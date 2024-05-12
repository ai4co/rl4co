from typing import Tuple

import torch.nn as nn

from torch import Tensor

from rl4co.models.common import ImprovementEncoder
from rl4co.models.nn.improvement_attention import (
    MultiHeadCompat,
    N2SEncoderLayer,
    mySequential,
)


class N2SEncoder(ImprovementEncoder):
    """Neural Neighborhood Search Encoder as in Ma et al. (2022)
    First embed the input and then process it with a Graph AttepdN2ntion Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the node embeddings
        pos_embedding: Module to use for the initialization of the positional embeddings
        env_name: Name of the environment used to initialize embeddings
        pos_type: Name of the used positional encoding method (CPE or APE)
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        pos_embedding: nn.Module = None,
        env_name: str = "pdp_ruin_repair",
        pos_type: str = "CPE",
        num_heads: int = 4,
        num_layers: int = 3,
        normalization: str = "layer",
        feedforward_hidden: int = 128,
    ):
        super(N2SEncoder, self).__init__(
            embed_dim=embed_dim,
            env_name=env_name,
            pos_type=pos_type,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
        )

        assert self.env_name in ["pdp_ruin_repair"], NotImplementedError()

        self.pos_net = MultiHeadCompat(num_heads, embed_dim, feedforward_hidden)

        self.net = mySequential(
            *(
                N2SEncoderLayer(
                    num_heads,
                    embed_dim,
                    feedforward_hidden,
                    normalization,
                )
                for _ in range(num_layers)
            )
        )

    def _encoder_forward(self, init_h: Tensor, init_p: Tensor) -> Tuple[Tensor, Tensor]:
        embed_p = self.pos_net(init_p)
        final_h, final_p = self.net(init_h, embed_p)

        return final_h, final_p
