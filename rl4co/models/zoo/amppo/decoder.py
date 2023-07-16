from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.utils import decode_probs, get_log_likelihood
from rl4co.models.zoo.common.autoregressive import AutoregressiveDecoder


class PPODecoder(AutoregressiveDecoder):
    def evaluate_action(
        self,
        td: TensorDict,
        embeddings: Tensor,
        action: Tensor,
        env: Union[str, RL4COEnvBase] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate the (old) action to compute
        log likelihood of the actions and corresponding entropy

        Args:
            td: Input TensorDict containing the environment state
            embeddings: Precomputed embeddings for the nodes
            action: Action to evaluate
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
        Returns:
            log_p: Tensor of shape (batch_size, seq_len, num_nodes) containing the log-likehood of the actions
            entropy: Tensor of shape (batch_size, seq_len) containing the sampled actions
        """

        log_p = []

        # Instantiate environment if needed
        if isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute(embeddings)

        decode_step = 0
        while not td["done"].all():
            log_p_, _ = self._get_log_p(cached_embeds, td)
            action_ = action[..., decode_step]

            td.set("action", action_)
            td = env.step(td)["next"]
            log_p.append(log_p_)

            decode_step += 1

        if len(log_p) != action.shape[-1]:
            raise ValueError(
                f"Action has {action.shape[1]} steps, but we decoded {decode_step} steps."
            )

        # Compute log likelihood of the actions
        log_p = torch.stack(log_p, 1)  # [batch_size, decoding steps, num_nodes]
        ll = get_log_likelihood(
            log_p, action, mask=None, return_sum=False
        )  # [batch_size, decoding steps]
        assert ll.isfinite().all(), "Log p is not finite"

        # compute entropy
        log_p = torch.nan_to_num(log_p, nan=0.0)
        entropy = -(log_p.exp() * log_p).sum(dim=-1)  # [batch, decoder steps]
        entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
        assert entropy.isfinite().all(), "Entropy is not finite"

        return ll, entropy


# class _PPODecoder(Decoder):

#     """
#     A slightly modified AM decoder to support PPO training.
#     """

#     def forward(
#         self,
#         td,
#         embeddings,
#         decode_type="sampling",
#         softmax_temp=None,
#         calc_reward: bool = True,
#         given_actions: torch.Tensor = None,  # [batch_size, graph_size]
#     ):
#         outputs = []
#         actions = []

#         # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
#         cached_embeds = self._precompute(embeddings)

#         decode_step = 0
#         while not td["done"].all():
#             log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp)

#             # Select the indices of the next nodes in the sequences, result (batch_size) long

#             if given_actions is not None:
#                 action = given_actions[..., decode_step]
#             else:
#                 action = decode_probs(log_p.exp(), mask, decode_type=decode_type)

#             td.set("action", action)
#             td = self.env.step(td)["next"]

#             outputs.append(log_p)
#             actions.append(action)

#             decode_step += 1

#         if given_actions is not None:
#             if len(outputs) != given_actions.shape[1]:
#                 # print(given_actions.shape, decode_step)
#                 # print(td["done"].all())
#                 raise ValueError(
#                     f"Given actions have {given_actions.shape[1]} steps, but we decoded {decode_step} steps."
#                 )

#         # output: logprobs [batch, problem size, decoding steps]
#         outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
#         if calc_reward:
#             td.set("reward", self.env.get_reward(td, actions))

#         return outputs, actions, td
