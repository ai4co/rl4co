import sys; sys.path.append('.')
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
    ):
        super(GCNEncoder, self).__init__()
        # Define the init embedding
        self.init_embedding = env_init_embedding(
            env, {"embedding_dim": embedding_dim}
        )

        # Generate edge index for a fully connected graph
        adj_matrix = torch.ones(num_nodes, num_nodes)
        adj_matrix.fill_diagonal_(0) # No self-loops
        self.edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0)).to(env.device)
        
        # Define the GCN layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        node_feature = self.init_embedding(x)

        # Create the batched graph
        data_list = [
            Data(x=x, edge_index=self.edge_index)
            for x in node_feature
        ]
        data_batch = Batch.from_data_list(data_list)

        update_node_feature = self.conv1(data_batch.x, self.edge_index)
        update_node_feature = F.relu(update_node_feature)
        update_node_feature = F.dropout(update_node_feature, training=self.training)
        update_node_feature = self.conv2(update_node_feature, self.edge_index)
        update_node_feature = F.log_softmax(update_node_feature, dim=-1)

        # De-batch the graph
        input_size = node_feature.size()
        update_node_feature = update_node_feature.view(*input_size)

        return update_node_feature, node_feature


if __name__ == '__main__':
    from rl4co.envs import TSPEnv
    env = TSPEnv()
    model = GCNEncoder(
        env=env,
        embedding_dim=128, 
        num_nodes=20,
    )
    td = env.reset(batch_size=[32])
    update_node_feature, _ = model(td)
    print(update_node_feature.size())