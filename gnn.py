import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GNNLayer(nn.Module):
    """Graph Neural Network Layer for processing graph structures.

    Args:
        units (int): The number of units in each linear transformation layer.
        act_fn (str): The name of the activation function to use after each linear layer. Defaults to 'silu'.
        agg_fn (str): The name of the global aggregation function to use for pooling features across the graph. Defaults to 'mean'.
    """

    def __init__(self, units, act_fn="silu", agg_fn="mean"):
        super(GNNLayer, self).__init__()
        self.units = units
        self.act_fn = getattr(nn.functional, act_fn)
        self.agg_fn = getattr(gnn, f"global_{agg_fn}_pool")

        # Vertex updates
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.v_bn = gnn.BatchNorm(units)

        # Edge updates
        self.e_lin = nn.Linear(units, units)
        self.e_bn = gnn.BatchNorm(units)

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x0 = x
        w0 = w

        # Vertex updates
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)
        x = x0 + self.act_fn(
            self.v_bn(
                x1 + self.agg_fn(torch.sigmoid(w0) * x2[edge_index[1]], edge_index[0])
            )
        )

        # Edge updates
        w1 = self.e_lin(w0)
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w


class GNNEncoder(nn.Module):
    """TGraph Neural Network for encoding graph structures into vector representations.

    Args:
        num_layers (int): The number of GNN layers to stack in the network.
        embedding_dim (int): The dimensionality of the embeddings for each node in the graph.
        act_fn (str): The activation function to use in each GNNLayer. Defaults to 'silu'.
        agg_fn (str): The aggregation function to use in each GNNLayer for pooling features. Defaults to 'mean'.
    """

    def __init__(self, num_layers, embedding_dim, act_fn="silu", agg_fn="mean"):
        super(GNNEncoder, self).__init__()
        self.act_fn = act_fn
        self.agg_fn = agg_fn

        # Stack of GNN layers
        self.layers = nn.ModuleList(
            [GNNLayer(embedding_dim, act_fn, agg_fn) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, w):
        """Sequentially passes the input graph data through the stacked GNN layers,
        applying specified transformations and aggregations to learn graph representations.

        Args:
            x (Tensor): The node features of the graph with shape [num_nodes, embedding_dim].
            edge_index (LongTensor): The edge indices of the graph with shape [2, num_edges].
            w (Tensor): The edge attributes or weights with shape [num_edges, num_edge_features].
        """
        for layer in self.layers:
            x, w = layer(x, edge_index, w)
        return x, w
