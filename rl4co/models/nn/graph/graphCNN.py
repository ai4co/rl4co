import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(
        self,
        env_name: str,
        embedding_dim,
        num_layers,
        init_embedding: nn.Module = None,
    ):
        """
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        """

        super(GraphCNN, self).__init__()
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if init_embedding is None
            else init_embedding
        )

        self.num_layers = num_layers

        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            self.mlps.append(MLP(embedding_dim, embedding_dim, num_neurons=[]))
            self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
        self.mlps.append(MLP(embedding_dim, embedding_dim, num_neurons=[]))
        self.batch_norms.append(nn.Identity())

    def next_layer(self, h, layer, adj_block=None):
        pooled = torch.bmm(adj_block, h)
        degree = adj_block.sum(2, keepdims=True)
        pooled = pooled / degree

        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep.view(-1, pooled_rep.size(-1))).view(
            *pooled_rep.size()
        )

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, td):
        # Transfer to embedding space
        init_h = self.init_embedding(td)
        x = init_h.clone()
        for layer in range(self.num_layers):
            x = self.next_layer(x, layer, adj_block=td["ops_on_same_ma_adj"])

        # x_nodes = x.clone()
        # # print(graph_pool.shape, h.shape)
        # pooled_x = torch.sparse.mm(graph_pool, x)
        # # pooled_h = graph_pool.spmm(h)

        # return pooled_x, x_nodes
        return x, init_h
