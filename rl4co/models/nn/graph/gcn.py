import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

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
        embedding_dim: int,
        num_nodes: int,
        num_gcn_layer: int,
        self_loop: bool = False,
        residual: bool = True,
    ):
        super(GCNEncoder, self).__init__()

        # Generate edge index for a fully connected graph
        adj_matrix = torch.ones(num_nodes, num_nodes)
        if self_loop:
            adj_matrix.fill_diagonal_(0)  # No self-loops
        self.edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))

        # Define the GCN layers
        self.gcn_layers = nn.ModuleList(
            [GCNConv(embedding_dim, embedding_dim) for _ in range(num_gcn_layer)]
        )

        # Record parameters
        self.residual = residual
        self.self_loop = self_loop

    def forward(self, x, node_feature, mask=None):
        """Forward pass of the GCN encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            node_feature: [batch_size, graph_size, embed_dim] node features, i.e. raw ones
            mask: [batch_size, graph_size] mask for valid nodes
        """

        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features

        # Check to update the edge index with different number of node
        if node_feature.size(1) != self.edge_index.max().item() + 1:
            adj_matrix = torch.ones(x.size(1), x.size(1))
            if self.self_loop:
                adj_matrix.fill_diagonal_(0)
            edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0))
        else:
            edge_index = self.edge_index

        # Create the batched graph
        data_list = [Data(x=x, edge_index=edge_index) for x in node_feature]
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
        input_size = node_feature.size()
        update_node_feature = update_node_feature.view(*input_size)

        # Residual
        update_node_feature = update_node_feature + node_feature

        return update_node_feature, node_feature


if __name__ == "__main__":
    from rl4co.envs import TSPEnv

    env = TSPEnv()
    model = GCNEncoder(
        env=env,
        embedding_dim=128,
        num_nodes=20,
        num_gcn_layer=3,
    )
    td = env.reset(batch_size=[32])
    update_node_feature, _ = model(td)
    print(update_node_feature.size())
