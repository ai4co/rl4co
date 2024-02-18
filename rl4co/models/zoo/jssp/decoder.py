import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.graph.graphCNN import GraphCNN
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.utils import decode_probs


class Decoder(nn.Module):
    # TODO GraphCNN + actor
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        tanh_exploration: float = 10.0,
        use_tanh: bool = True,
        num_glimpses=1,
        mask_glimpses=True,
        mask_logits=True,
    ):
        super(Decoder, self).__init__()

        self.graph_cnn = GraphCNN(
            num_layers=2,
            num_mlp_layers=2,
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            learn_eps=False,
            neighbor_pooling_type="sum",
            device=self.device,
        )

        self.actor = MLP(
            input_dim=hidden_dim,
            output_dim=1,
            num_neurons=[hidden_dim, hidden_dim],
            hidden_act="tanh",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):
        logit_mask = (
            self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        )

        logits, h_out = self.calc_logits(
            x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits
        )

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:
            probs[logit_mask] = 0.0

        return h_out, log_p, probs, logit_mask

    def calc_logits(
        self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None
    ):
        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.num_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = float("-inf")
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
            # [batch_size x h_dim x 1]
            g_l = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = float("-inf")

        return logits, h_out

    def forward(
        self,
        decoder_input,
        embedded_inputs,
        hidden,
        context,
        decode_type="sampling",
        eval_tours=None,
    ):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = torch.zeros(
            embedded_inputs.size(1),
            embedded_inputs.size(0),
            dtype=torch.bool,
            device=embedded_inputs.device,
        )

        for i in steps:
            hidden, log_p, probs, mask = self.recurrence(
                decoder_input, hidden, mask, idxs, i, context
            )
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = (
                decode_probs(probs, mask, decode_type=decode_type)
                if eval_tours is None
                else eval_tours[:, i]
            )

            idxs = (
                idxs.detach()
            )  # Otherwise pytorch complains it want's a reward, todo implement this more properly?

            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous()
                .view(1, batch_size, 1)
                .expand(1, batch_size, *embedded_inputs.size()[2:]),
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)
        return (torch.stack(outputs, 1), torch.stack(selections, 1)), hidden
