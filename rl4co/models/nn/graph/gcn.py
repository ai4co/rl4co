from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict
from torch import Tensor
from torch.nn.modules.module import Module
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv as PygGCNConv

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.utils.ops import get_full_graph_edge_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


EdgeIndexFnSignature = Callable[[TensorDict, int, bool], Tensor]


def edge_idx_fn_wrapper(td: TensorDict, num_nodes: int, self_loop: bool):
    return get_full_graph_edge_index(td.device, num_nodes, self_loop)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907.
    Taken from https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py
    but refactored to work for batches of data
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1.0 / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


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
        num_layers: int,
        init_embedding: nn.Module = None,
        dropout: float = 0.5,
        residual: bool = True,
        adj_key: str = "adjacency",
        self_loop: bool = True,
    ):
        super(GCNEncoder, self).__init__()

        self.env_name = env_name
        self.embedding_dim = embedding_dim
        self.adj_key = adj_key
        self.dropout = dropout
        self.residual = residual
        self.self_loop = self_loop

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if init_embedding is None
            else init_embedding
        )

        # Define the GCN layers
        self.gcn_layers = nn.ModuleList(
            [GraphConvolution(embedding_dim, embedding_dim) for _ in range(num_layers)]
        )

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space; shape=(bs, num_nodes, emb_dim)
        init_h = self.init_embedding(td)
        num_nodes = init_h.size(1)
        # prepare data for gcn
        update_node_feature = init_h.clone()
        adj = td[self.adj_key]

        if self.self_loop:
            self_loop = torch.eye(num_nodes, device=td.device, dtype=torch.bool)
            adj = adj.masked_fill(self_loop, 1)

        # Symmetric normalization of the adjacency matrix; shape=(bs, num_nodes)
        row_sum = torch.sum(adj, dim=-1)
        # shape=(bs, num_nodes, num_nodes)
        D_inv_sqrt = torch.diag_embed(torch.pow(row_sum, -0.5))
        # shape=(bs, num_nodes, num_nodes)
        adj_norm = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)

        # GCN process
        for layer in self.gcn_layers[:-1]:
            update_node_feature = layer(update_node_feature, adj_norm)
            update_node_feature = F.relu(update_node_feature)
            update_node_feature = F.dropout(update_node_feature, training=self.training)

        # last layer without relu activation and dropout
        update_node_feature = self.gcn_layers[-1](update_node_feature, adj_norm)

        # Residual
        if self.residual:
            update_node_feature = update_node_feature + init_h

        return update_node_feature, init_h


class PygGCNEncoder(nn.Module):
    """Graph Convolutional Network to encode embeddings with a series of GCN layers
    from the pytorch geometric package

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
        num_layers: int,
        init_embedding: nn.Module = None,
        self_loop: bool = True,
        residual: bool = True,
        edge_idx_fn: EdgeIndexFnSignature = None,
    ):
        super(PygGCNEncoder, self).__init__()

        self.env_name = env_name
        self.embedding_dim = embedding_dim

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if init_embedding is None
            else init_embedding
        )

        if edge_idx_fn is None:
            log.warning("No edge indices passed. Assume a fully connected graph")
            edge_idx_fn = edge_idx_fn_wrapper

        self.edge_idx_fn = edge_idx_fn

        # Define the GCN layers
        self.gcn_layers = nn.ModuleList(
            [PygGCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)]
        )

        # Record parameters
        self.residual = residual
        self.self_loop = self_loop

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

        # Create the batched graph
        # TODO this is extremely inefficient and should probably be moved outside the forward pass
        data_list = [
            Data(x=x, edge_index=self.edge_idx_fn(td[i], num_node, self.self_loop))
            for i, x in enumerate(init_h)
        ]
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
        if self.residual:
            update_node_feature = update_node_feature + init_h

        return update_node_feature, init_h
