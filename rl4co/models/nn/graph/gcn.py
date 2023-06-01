import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GCNEncoder(nn.Module):
    def __init__(
        self,
        env,
        embedding_dim,
        num_nodes,
        num_gcn_layer,
        self_loop=False,
        residual=True,
    ):
        super(GCNEncoder, self).__init__()
        # Define the init embedding
        self.init_embedding = env_init_embedding(env, {"embedding_dim": embedding_dim})

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

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        node_feature = self.init_embedding(x)

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
        update_node_feature = F.log_softmax(update_node_feature, dim=-1)

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
