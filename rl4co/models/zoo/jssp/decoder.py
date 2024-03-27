import torch
import torch.nn as nn

from rl4co.models.nn.graph.graphCNN import GraphCNN
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.utils import decode_probs


class Decoder(nn.Module):
    # TODO GraphCNN + actor
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
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

    def forward(
        self,
        td,
        decode_type="sampling",
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

        td_out = self.graph_cnn(td)
        td_out = self.actor(td_out["embeddings"])
        log_p = torch.log_softmax(td_out["logits"], dim=1)
        probs = log_p.exp()
        actions = decode_probs(probs, td["mask"], decode_type=decode_type)

        return log_p, actions, td_out

    def evaluate_action(td, action, env):
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        log_p = []
        decode_step = 0
        while not td["done"].all():
            log_p_, _ = self._get_log_p(cached_embeds, td)
            action_ = action[..., decode_step]

            td.set("action", action_)
            td = env.step(td)["next"]
            log_p.append(log_p_)

            decode_step += 1

        # Note that the decoding steps may not be equal to the decoding steps of actions
        # due to the padded zeros in the actions

        # Compute log likelihood of the actions
        log_p = torch.stack(log_p, 1)  # [batch_size, decoding steps, num_nodes]
        ll = get_log_likelihood(
            log_p, action[..., :decode_step], mask=None, return_sum=False
        )  # [batch_size, decoding steps]
        assert ll.isfinite().all(), "Log p is not finite"

        # compute entropy
        log_p = torch.nan_to_num(log_p, nan=0.0)
        entropy = -(log_p.exp() * log_p).sum(dim=-1)  # [batch, decoder steps]
        entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
        assert entropy.isfinite().all(), "Entropy is not finite"

        return ll, entropy
