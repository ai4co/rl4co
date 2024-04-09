from functools import lru_cache
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

try:
    from torch_geometric.data import Batch
except ImportError:
    # `Batch` is referred to only as type notations in this file
    Batch = None

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import DecodingStrategy, get_decoding_strategy
from rl4co.utils.ops import batchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class EdgeHeatmapGenerator(nn.Module):
    """MLP for converting edge embeddings to heatmaps

    Args:
        embedding_dim: Dimension of the embeddings
        num_layers: The number of linear layers in the network.
        act_fn: Activation function. Defaults to "silu".
        linear_bias: Use bias in linear layers. Defaults to True.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        act_fn: Union[str, Callable] = "silu",
        linear_bias: bool = True,
        undirected_graph: bool = True,
    ) -> None:
        super(EdgeHeatmapGenerator, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)
                for _ in range(num_layers - 1)
            ]
        )
        self.output = nn.Linear(embedding_dim, 1, bias=linear_bias)

        self.act = getattr(nn.functional, act_fn) if isinstance(act_fn, str) else act_fn

        self.undirected_graph = undirected_graph

    def forward(self, graph: Batch) -> Tensor:  # type: ignore
        # do not reuse the input value
        edge_attr = graph.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        graph.edge_attr = torch.sigmoid(self.output(edge_attr)) * 10  # type: ignore

        heatmaps_logits = self._make_heatmaps(graph)
        return heatmaps_logits

    def _make_heatmaps(self, batch_graph: Batch) -> Tensor:  # type: ignore
        graphs = batch_graph.to_data_list()
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].x.shape[0]

        heatmaps_logits = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            device=device,
            dtype=graphs[0].edge_attr.dtype,
        )

        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmaps_logits[index, edge_index[0], edge_index[1]] = edge_attr.flatten()

        if self.undirected_graph:
            heatmaps_logits = (heatmaps_logits + heatmaps_logits.transpose(1, 2)) * 0.5

        return heatmaps_logits


class NonAutoregressiveDecoder(nn.Module):
    """Non-autoregressive decoder for constructing solutions for combinatorial optimization problems.
    This model utilizes a multi-layer perceptron (MLP) approach to predict edge attributes directly from the input graph features,
    which are then transformed into a heatmap representation to facilitate the decoding of the solution. The decoding process
    is managed by a specified strategy which could vary from simple greedy selection to more complex sampling methods.

    Note:
        This decoder's performance heavily relies on the ability of the MLP to capture the dependencies between different
        parts of the solution without the iterative refinement provided by autoregressive models. It is particularly useful
        in scenarios where the solution space can be effectively explored in a parallelized manner or when the solution components
        are largely independent.

    Warning:
        The effectiveness of the non-autoregressive approach can vary significantly across different problem types and configurations.
        It may require careful tuning of the model architecture and decoding strategy to achieve competitive results.

    Args:
        env_name: environment name to solve
        embedding_dim: Dimension of the embeddings
        num_layers: Number of linear layers to use in the MLP
        act_fn: Activation function to use between linear layers. Can be a string name or a direct callable
        linear_bias: Whether to use a bias term in the linear layers
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int,
        num_layers: int,
        heatmap_generator: Optional[nn.Module] = None,
        linear_bias: bool = True,
    ) -> None:
        super(NonAutoregressiveDecoder, self).__init__()

        self.env_name = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        if heatmap_generator is None:
            self.heatmap_generator = EdgeHeatmapGenerator(
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                linear_bias=linear_bias,
            )
        else:
            self.heatmap_generator = heatmap_generator

    def forward(
        self,
        td: TensorDict,
        graph: Batch,  # type: ignore
        env: Union[str, RL4COEnvBase, None] = None,
        decode_type: str = "multistart_sampling",
        calc_reward: bool = True,
        phase="train",
        **strategy_kwargs,
    ):
        # Instantiate environment if needed
        if env is None or isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # calculate heatmap
        heatmaps_logits = self.heatmap_generator(graph)

        # setup decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type, **strategy_kwargs
        )
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Main decoding: loop until all sequences are done
        while not td["done"].all():
            logits, mask = self._get_logits(td, heatmaps_logits, num_starts)
            td = decode_strategy.step(logits, mask, td)
            td = env.step(td)["next"]

        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        return logprobs, actions, td

    @classmethod
    def _get_logits(cls, td: TensorDict, heatmaps_logits: Tensor, num_starts: int):
        current_action = td.get("action", None)
        if current_action is None:
            logits = heatmaps_logits.mean(-1)
        else:
            batch_size = heatmaps_logits.shape[0]
            _indexer = cls._multistart_batched_index(batch_size, num_starts)
            logits = heatmaps_logits[_indexer, current_action, :]
        return logits, td["action_mask"]

    @staticmethod
    @lru_cache(10)
    def _multistart_batched_index(batch_size: int, num_starts: int):
        arr = torch.arange(batch_size)
        if num_starts <= 1:
            return arr
        else:
            return batchify(arr, num_starts)
