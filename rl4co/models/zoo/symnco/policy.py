from typing import Union

import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.modules.models import MLP

from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SymNCOPolicy(AutoregressivePolicy):
    """SymNCO Policy based on AutoregressivePolicy.
    This differs from the default :class:`AutoregressivePolicy` in that it
    projects the initial embeddings to a lower dimension using a projection head and
    returns it. This is used in the SymNCO algorithm to compute the invariance loss.
    Based on Kim et al. (2022) https://arxiv.org/abs/2205.13209.

    Args:
        env_name: Name of the environment
        embedding_dim: Dimension of the embedding
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the encoder
        normalization: Normalization to use in the encoder
        projection_head: Projection head to use
        use_projection_head: Whether to use projection head
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        projection_head: nn.Module = None,
        use_projection_head: bool = True,
        **kwargs,
    ):
        super(SymNCOPolicy, self).__init__(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )

        self.use_projection_head = use_projection_head

        if self.use_projection_head:
            self.projection_head = (
                MLP(embedding_dim, embedding_dim, 1, embedding_dim, nn.ReLU)
                if projection_head is None
                else projection_head
            )

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = True,
        **decoder_kwargs,
    ) -> dict:
        super().forward.__doc__  # trick to get docs from parent class

        # Ensure that if use_projection_head is True, then return_init_embeds is True
        assert not (
            self.use_projection_head and not return_init_embeds
        ), "If `use_projection_head` is True, then we must `return_init_embeds`"

        out = super().forward(
            td,
            env,
            phase,
            return_actions,
            return_entropy,
            return_init_embeds,
            **decoder_kwargs,
        )

        # Project initial embeddings
        if self.use_projection_head:
            out["proj_embeddings"] = self.projection_head(out["init_embeds"])

        return out
