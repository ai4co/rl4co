from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    Batch = Data = None

from rl4co.utils.ops import get_distance_matrix, get_full_graph_edge_index, sparsify_graph
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_edge_embedding(env_name: str, config: dict) -> nn.Module:
    """Retrieve the edge embedding module specific to the environment. Edge embeddings are crucial for
    transforming the raw edge features into a format suitable for the neural network, especially in
    graph neural networks where edge features can significantly impact the model's performance.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPEdgeEmbedding,
        "atsp": ATSPEdgeEmbedding,
        "cvrp": TSPEdgeEmbedding,
        "cvrpmvc": TSPEdgeEmbedding,
        "sdvrp": TSPEdgeEmbedding,
        "pctsp": TSPEdgeEmbedding,
        "spctsp": TSPEdgeEmbedding,
        "op": TSPEdgeEmbedding,
        "dpp": TSPEdgeEmbedding,
        "mdpp": TSPEdgeEmbedding,
        "pdp": TSPEdgeEmbedding,
        "mtsp": TSPEdgeEmbedding,
        "smtwtp": NoEdgeEmbedding,
        "shpp": TSPEdgeEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPEdgeEmbedding(nn.Module):
    """Edge embedding module for the Traveling Salesman Problem (TSP) and related problems.
    This module converts the cost matrix or the distances between nodes into embeddings that can be
    used by the neural network. It supports sparsification to focus on a subset of relevant edges,
    which is particularly useful for large graphs.
    """

    node_dim = 1

    def __init__(
        self,
        embed_dim,
        linear_bias=True,
        sparsify=True,
        k_sparse: Optional[int] = None,
    ):
        assert Batch is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

        super(TSPEdgeEmbedding, self).__init__()
        self.k_sparse = k_sparse
        self.sparsify = sparsify
        self.edge_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)

    def forward(self, td, init_embeddings: Tensor):
        cost_matrix = get_distance_matrix(td["locs"])
        batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
        return batch

    def _cost_matrix_to_graph(self, batch_cost_matrix: Tensor, init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n]
            init_embedding: init embeddings
        """
        graph_data = []
        for index, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(
                    cost_matrix, self.k_sparse, self_loop=False
                )
            else:
                edge_index = get_full_graph_edge_index(
                    cost_matrix.shape[0], self_loop=False
                ).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]]

            graph = Data(
                x=init_embeddings[index],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )  # type: ignore
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch


class VRPPolarEdgeEmbedding(TSPEdgeEmbedding):
    """TODO"""

    node_dim = 2

    def forward(self, td, init_embeddings: Tensor):
        with torch.no_grad():
            if "polar_locs" in td.keys():
                theta = td["polar_locs"][..., 1]
            else:
                shifted_locs = td["locs"] - td["locs"][..., 0:1, :]
                x, y = shifted_locs[..., 0], shifted_locs[..., 1]
                theta = torch.atan2(y, x)

            delta_theta_matrix = theta[..., :, None] - theta[..., None, :]
            edge_attr1 = 1 - torch.cos(delta_theta_matrix)
            edge_attr2 = get_distance_matrix(td["locs"])
            cost_matrix = torch.stack((edge_attr1, edge_attr2), dim=-1)
            del edge_attr1, edge_attr2, delta_theta_matrix

            batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
            del cost_matrix

        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch

    @torch.no_grad()
    def _cost_matrix_to_graph(self, batch_cost_matrix: Tensor, init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n, m]
            init_embedding: init embeddings of shape [batch_size, n, m]
        """
        graph_data = []
        k_sparse = self.get_k_sparse(batch_cost_matrix.shape[2])

        for index, cost_matrix in enumerate(batch_cost_matrix):
            edge_index, _ = sparsify_graph(cost_matrix[..., 0], k_sparse, self_loop=False)
            edge_index = edge_index.T[torch.all(edge_index != 0, dim=0)].T
            _, depot_edge_index = torch.topk(
                cost_matrix[0, :, 1], k=k_sparse, largest=False, sorted=False
            )
            depot_edge_index = depot_edge_index[depot_edge_index != 0]
            depot_edge_index = torch.stack(
                (torch.zeros_like(depot_edge_index), depot_edge_index), dim=0
            )
            edge_index = torch.concat((depot_edge_index, edge_index), dim=-1).detach()
            edge_attr = cost_matrix[edge_index[0], edge_index[1]].detach()

            graph = Data(
                x=init_embeddings[index],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )  # type: ignore
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        return batch

    def get_k_sparse(self, n_nodes):
        # for reproducing GLOP
        if self.k_sparse is None:
            if n_nodes >= 30000:
                k_sparse = 350
            elif n_nodes >= 10000:
                k_sparse = 300
            elif n_nodes >= 7000:
                k_sparse = 250
            elif n_nodes >= 3000:
                k_sparse = 200
            else:
                k_sparse = min(100, max(n_nodes // 10, 10))
            return k_sparse
        else:
            return self.k_sparse


class ATSPEdgeEmbedding(TSPEdgeEmbedding):
    """Edge embedding module for the Asymmetric Traveling Salesman Problem (ATSP).
    Inherits from TSPEdgeEmbedding and adapts the edge embedding process to handle
    asymmetric cost matrices, where the cost from node i to node j may not be the same as from j to i.
    """

    def forward(self, td, init_embeddings: Tensor):
        batch = self._cost_matrix_to_graph(td["cost_matrix"], init_embeddings)
        return batch


class NoEdgeEmbedding(nn.Module):
    """A module for environments that do not require edge embeddings, or where edge features
    are not used. This can be useful for simplifying models in problems where only node
    features are relevant.
    """

    def __init__(self, embed_dim, self_loop=False, **kwargs):
        assert Batch is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

        super(NoEdgeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.self_loop = self_loop

    def forward(self, td, init_embeddings: Tensor):
        data_list = []
        n = init_embeddings.shape[1]
        device = init_embeddings.device
        edge_index = get_full_graph_edge_index(n, self_loop=self.self_loop).to(device)
        m = edge_index.shape[1]

        for node_embed in init_embeddings:
            data = Data(
                x=node_embed,
                edge_index=edge_index,
                edge_attr=torch.zeros((m, self.embed_dim), device=device),
            )  # type: ignore
            data_list.append(data)

        batch = Batch.from_data_list(data_list)  # type: ignore
        return batch
