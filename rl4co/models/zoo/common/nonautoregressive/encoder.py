from typing import Optional, Union

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.env_embeddings import env_edge_embedding, env_init_embedding
from rl4co.models.nn.graph.gnn import GNNEncoder


class NonAutoregressiveEncoder(nn.Module):
    """Anisotropic Graph Neural Network encoder with edge-gating mechanism as in Joshi et al. (2022)

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_layers: Number of layers in the encoder
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        act_fn: The activation function to use in each GNNLayer, see https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions for available options. Defaults to 'silu'.
        agg_fn: The aggregation function to use in each GNNLayer for pooling features. Options: 'add', 'mean', 'max'. Defaults to 'mean'.
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
        super(NonAutoregressiveEncoder, self).__init__()
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

    def forward(self, td: TensorDict):
        """Forward pass of the encoder.
        Transform the input TensorDict into the latent representation.
        """
        # Transfer to embedding space
        node_embed = self.init_embedding(td)
        graph = self.edge_embedding(td, node_embed)

        # Process embedding
        graph.x, graph.edge_attr = self.net(graph.x, graph.edge_index, graph.edge_attr)

        # Return latent representation and initial embeddings
        return graph, node_embed
