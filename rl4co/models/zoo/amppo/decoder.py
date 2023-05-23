import torch

from rl4co.models.nn.utils import decode_probs
from rl4co.models.zoo.am.decoder import Decoder


class PPODecoder(Decoder):

    """
    A slightly modified AM decoder to support PPO training.
    """

    def forward(
        self,
        td,
        embeddings,
        decode_type="sampling",
        softmax_temp=None,
        calc_reward: bool = True,
        given_actions: torch.Tensor = None,  # [batch_size, graph_size]
    ):
        outputs = []
        actions = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute(embeddings)

        decode_step = 0
        while not td["done"].all():
            log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp)

            # Select the indices of the next nodes in the sequences, result (batch_size) long

            if given_actions is not None:
                action = given_actions[:, decode_step]
            else:
                action = decode_probs(log_p.exp(), mask, decode_type=decode_type)

            td.set("action", action)
            td = self.env.step(td)["next"]

            # Collect output of step
            # print("log_p", log_p.shape)
            # raise RuntimeError("stop")

            outputs.append(log_p)
            actions.append(action)

            decode_step += 1

        # output: logprobs [batch, problem size, decoding steps]
        outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        if calc_reward:
            td.set("reward", self.env.get_reward(td, actions))

        return outputs, actions, td
