from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GCNEncoder(nn.Module):
    """Graph Convolutional Network to encode embeddings with a series of GCN layers

    Args:
        embedding_dim: dimension of the embeddings
        num_nodes: number of nodes in the graph
        num_gcn_layer: number of GCN layers
        self_loop: whether to add self loop in the graph
        residual: whether to use residual connection
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int,
        num_nodes: int,
        num_layers: int,
        init_embedding: nn.Module = None,
        self_loop: bool = False,
        residual: bool = True,
    ):
        super(GCNEncoder, self).__init__()

        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if init_embedding is None
            else init_embedding
        )

        # Generate edge index for a fully connected graph
        adj_matrix = torch.ones(num_nodes, num_nodes)
        if self_loop:
            adj_matrix.fill_diagonal_(0)  # No self-loops
        self.edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))

        # Define the GCN layers
        self.gcn_layers = nn.ModuleList(
            [GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)]
        )

        # Record parameters
        self.residual = residual
        self.self_loop = self_loop

    # def forward(self, x, node_feature, mask=None):
    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)
        num_node = init_h.size(-2)

        # Check to update the edge index with different number of node
        if num_node != self.edge_index.max().item() + 1:
            adj_matrix = torch.ones(num_node, num_node)
            if self.self_loop:
                adj_matrix.fill_diagonal_(0)
            edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))
            edge_index = edge_index.to(init_h.device)
        else:
            edge_index = self.edge_index.to(init_h.device)

        # Create the batched graph
        data_list = [Data(x=x, edge_index=edge_index) for x in init_h]
        data_batch = Batch.from_data_list(data_list)

        # GCN process
        update_node_feature = data_batch.x
        edge_index = data_batch.edge_index
        for layer in self.gcn_layers[:-1]:
            update_node_feature = layer(update_node_feature, edge_index)
            update_node_feature = F.relu(update_node_feature)
            update_node_feature = F.dropout(update_node_feature, training=self.training)

        update_node_feature = self.gcn_layers[-1](update_node_feature, edge_index)

        # De-batch the graph
        input_size = init_h.size()
        update_node_feature = update_node_feature.view(*input_size)

        # Residual
        update_node_feature = update_node_feature + init_h

        return update_node_feature, init_h
