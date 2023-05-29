import sys
from turtle import up; sys.path.append('.')
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch

from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.env_embedding import env_init_embedding

log = get_pylogger(__name__)


class MessagePassingEncoder(MessagePassing):
    def __init__(
        self,
        env,
        embedding_dim,
        num_nodes,
        aggregation="add",
        residual=False,
        **mlp_params,
    ):
        super(MessagePassingEncoder, self).__init__(aggr=aggregation)
        # Define the init embedding
        self.init_embedding = env_init_embedding(
            env, {"embedding_dim": embedding_dim}
        )

        # Generate edge index for a fully connected graph
        adj_matrix = torch.ones(num_nodes, num_nodes)
        adj_matrix.fill_diagonal_(0) # No self-loops
        self.edge_index = torch.permute(torch.nonzero(adj_matrix), (1, 0)).to(env.device)

        # Init message passing models
        edge_indim = 1 # Distance
        edge_outdim = 1
        node_indim = embedding_dim # Position
        node_outdim = embedding_dim
        self.edge_model = MLP(
            input_dim=edge_indim + 2 * node_indim,
            output_dim=edge_outdim,
            **mlp_params
        )
        self.node_model = MLP(
            input_dim=edge_outdim + node_indim,
            output_dim=node_outdim,
            **mlp_params
        )
        self.residual = residual

    def forward(self, x, mask=None):
        '''
        Args:
            - x <tensorDict>
        '''
        assert mask is None, "Mask not yet supported!"
        # Initialize embedding
        node_feature = self.init_embedding(x)

        # Generate edge features: distance
        edge_feature = torch.norm(
            node_feature[..., self.edge_index[0], :] - node_feature[..., self.edge_index[1], :],
            dim=-1,
            keepdim=True,
        )

        # Create the batched graph
        data_list = [
            Data(x=x, edge_index=self.edge_index, edge_attr=edge_attr)
            for x, edge_attr in zip(node_feature, edge_feature)
        ]
        data_batch = Batch.from_data_list(data_list)

        # Message passing
        update_edge_feature = self.edge_update(
            data_batch.x, data_batch.edge_attr, data_batch.edge_index
        )
        update_node_feature = self.propagate(
            data_batch.edge_index, x=data_batch.x, edge_features=update_edge_feature
        )

        # De-batch the graph
        input_size = node_feature.size()
        update_node_feature = update_node_feature.view(*input_size)

        # Update with residual connection
        if self.residual:
            update_node_feature = update_node_feature + node_feature

        return update_node_feature, node_feature

    def edge_update(self, nf, ef, edge_index):
        row, col = edge_index
        x_i, x_j = nf[row], nf[col]
        uef = self.edge_model(torch.cat([x_i, x_j, ef], dim=-1))
        return uef

    def message(self, edge_features: torch.tensor):
        return edge_features

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        unf = self.node_model(torch.cat([x, aggr_msg], dim=-1))
        return unf


if __name__ == '__main__':
    from rl4co.envs import TSPEnv
    env = TSPEnv()
    model = MessagePassingEncoder(
        env=env,
        embedding_dim=128, 
        num_nodes=20
    )
    td = env.reset(batch_size=[32])
    update_node_feature, _ = model(td)
    print(update_node_feature.size())