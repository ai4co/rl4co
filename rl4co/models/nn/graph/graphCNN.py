import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        learn_eps,
        neighbor_pooling_type,
        device,
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

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlp = torch.nn.ModuleList()
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        num_neurons = [hidden_dim] * num_mlp_layers
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(input_dim, hidden_dim, num_neurons))
            else:
                self.mlps.append(MLP(hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list=None, adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(
                    adj_block, torch.ones((adj_block.shape[0], 1)).to(self.device)
                )
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, adj_block=None):
        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            # print(adj_block.dtype)
            # print(h.dtype)
            pooled = torch.mm(adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(
                    adj_block, torch.ones((adj_block.shape[0], 1)).to(self.device)
                )
                pooled = pooled / degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self, x, graph_pool, padded_nei, adj):
        if self.neighbor_pooling_type == "max" and self.learn_eps:
            for layer in range(self.num_layers - 1):
                x = self.next_layer_eps(x, layer, padded_neighbor_list=padded_nei)
        elif not self.neighbor_pooling_type == "max" and self.learn_eps:
            for layer in range(self.num_layers - 1):
                x = self.next_layer_eps(x, layer, adj_block=adj)
        elif self.neighbor_pooling_type == "max" and not self.learn_eps:
            for layer in range(self.num_layers - 1):
                x = self.next_layer(x, layer, padded_neighbor_list=padded_nei)
        elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
            for layer in range(self.num_layers - 1):
                x = self.next_layer(x, layer, adj_block=adj)

        x_nodes = x.clone()
        # print(graph_pool.shape, h.shape)
        pooled_x = torch.sparse.mm(graph_pool, x)
        # pooled_h = graph_pool.spmm(h)

        return pooled_x, x_nodes
