from typing import Tuple, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.common.decoder_only.decoder import Decoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DecoderOnlyPolicy(nn.Module):
    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int = 256,
        feature_extractor: nn.Module = None,
        actor: nn.Module = None,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.decoder = Decoder(
            env_name=self.env_name,
            embedding_dim=embedding_dim,
            feature_extractor=feature_extractor,
            actor=actor,
        )

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        **decoder_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            decoder_kwargs: Keyword arguments for the decoder. See :class:`rl4co.models.zoo.common.autoregressive.decoder.AutoregressiveDecoder`

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # DECODER: main rollout with autoregressive decoding
        log_p, actions, td_out = self.decoder(td, env, **decoder_kwargs)

        # Log likelihood is calculated within the model
        log_likelihood = get_log_likelihood(log_p, actions, td_out.get("mask", None))

        out = {
            "reward": td_out["reward"],
            "log_likelihood": log_likelihood,
        }
        if return_actions:
            out["actions"] = actions

        if return_entropy:
            entropy = -(log_p.exp() * log_p).nansum(dim=1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy

        # env.render(td_out, 0)

        return out

    def evaluate_action(
        self,
        td: TensorDict,
        action: Tensor,
        env: Union[str, RL4COEnvBase] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate the action probability and entropy under the current policy

        Args:
            td: TensorDict containing the current state
            action: Action to evaluate
            env: Environment to evaluate the action in.
        """
        ll, entropy = self.decoder.evaluate_action(td, action, env)
        return ll, entropy
