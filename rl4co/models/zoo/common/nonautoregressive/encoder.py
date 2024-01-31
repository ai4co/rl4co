from typing import Optional, Union

import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.env_embeddings.edge import env_edge_embedding
from rl4co.models.nn.graph.gnn import GNNEncoder


class AnisotropicGNNEncoder(nn.Module):
    """Anisotropic Graph Neural Networks as in Joshi et al. (2022)

    Args:
        TODO
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int,
        num_layers: int,
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        act_fn="silu",
        agg_fn="mean",
    ):
        super(AnisotropicGNNEncoder, self).__init__()
        self.env_name = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if init_embedding is None
            else init_embedding
        )

        self.edge_embedding = (
            env_edge_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if edge_embedding is None
            else edge_embedding
        )

        self.net = GNNEncoder(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            act_fn=act_fn,
            agg_fn=agg_fn,
        )

    def forward(self, td: TensorDict) -> Tensor:
        """Forward pass of the encoder.
        Transform the input TensorDict into the latent representation.
        """
        # Transfer to embedding space
        node_embed = self.init_embedding(td)
        data = self.edge_embedding(td, node_embed)

        # Process embedding
        data.x, data.edge_attr = self.net(data.x, data.edge_index, data.edge_attr)

        # Return latent representation
        return data
