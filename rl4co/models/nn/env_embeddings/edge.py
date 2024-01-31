import torch
import torch.nn as nn

from torch import Tensor

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    raise ImportError(
        "torch_geometric not found. Please install torch_geometric using instructions from "
        "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html ."
    )

from rl4co.utils.ops import get_distance_matrix, get_full_graph_edge_index, sparsify_graph
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_edge_embedding(env_name: str, config: dict) -> nn.Module:
    """TODO

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPEdgeEmbedding,
        "atsp": ATSPEdgeEmbedding,
        "cvrp": TSPEdgeEmbedding,
        "sdvrp": TSPEdgeEmbedding,
        "pctsp": TSPEdgeEmbedding,
        "spctsp": TSPEdgeEmbedding,
        "op": TSPEdgeEmbedding,
        "dpp": TSPEdgeEmbedding,
        "mdpp": TSPEdgeEmbedding,
        "pdp": TSPEdgeEmbedding,
        "mtsp": TSPEdgeEmbedding,
        "smtwtp": NoEdgeEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPEdgeEmbedding(nn.Module):
    """TODO"""

    def __init__(
        self,
        embedding_dim,
        linear_bias=True,
        sparsify=True,
        k_sparse: int = None,
    ):
        super(TSPEdgeEmbedding, self).__init__()
        node_dim = 1
        self.k_sparse = k_sparse
        self.sparsify = sparsify
        self.edge_embed = nn.Linear(node_dim, embedding_dim, linear_bias)

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
            )
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch


class ATSPEdgeEmbedding(TSPEdgeEmbedding):
    """TODO"""

    def forward(self, td, init_embeddings: Tensor):
        batch = self._cost_matrix_to_graph(td["cost_matrix"], init_embeddings)
        return batch


class NoEdgeEmbedding(nn.Module):
    """TODO"""

    def __init__(self, embedding_dim, self_loop=False, **kwargs):
        super(NoEdgeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_loop = self_loop

    def forward(self, td, init_embeddings: Tensor):
        data_list = []
        n = init_embeddings.shape[1]
        device = init_embeddings.device
        edge_index = get_full_graph_edge_index(n, self_loop=self.self_loop).to(device)

        for node_embed in init_embeddings:
            data = Data(
                x=node_embed,
                edge_index=edge_index,
                edge_attr=torch.zeros((n, self.embedding_dim), device=device),
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        return batch
