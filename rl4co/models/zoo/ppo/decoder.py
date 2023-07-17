from typing import Tuple, Union

import torch
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.utils import get_log_likelihood
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
            action: Action to evaluate (batch_size, seq_len)
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
        Returns:
            log_p: Tensor of shape (batch_size, seq_len, num_nodes) containing the log-likehood of the actions
            entropy: Tensor of shape (batch_size, seq_len) containing the sampled actions
        """

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute_cache(embeddings)

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
