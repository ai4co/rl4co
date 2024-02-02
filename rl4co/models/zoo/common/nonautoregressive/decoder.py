from functools import lru_cache
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.dec_strategies import DecodingStrategy, get_decoding_strategy
from rl4co.utils.ops import batchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@lru_cache(10)  # cache the result to reduce computational redundency
def _multistart_batched_index(batch_size: int, num_starts: int):
    arr = torch.arange(batch_size)
    if num_starts <= 1:
        return arr
    else:
        return batchify(arr, num_starts)


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

    def forward(self, data: Batch) -> Tensor:
        # do not reuse the input value
        edge_attr = data.edge_attr  # type: ignore
        for layer in self.linears:
            edge_attr = self.act(layer(edge_attr))
        data.edge_attr = torch.sigmoid(self.output(edge_attr)) * 10  # type: ignore

        heatmaps_logp = self._make_heatmaps(data)
        return heatmaps_logp

    def _make_heatmaps(self, batch_graph: Batch) -> Tensor:
        graphs = batch_graph.to_data_list()
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].x.shape[0]

        heatmaps_logp = torch.zeros(
            (batch_size, num_nodes, num_nodes),
            device=device,
            dtype=graphs[0].edge_attr.dtype,
        )

        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmaps_logp[index, edge_index[0], edge_index[1]] = edge_attr.flatten()

        if self.undirected_graph:
            heatmaps_logp = (heatmaps_logp + heatmaps_logp.transpose(1, 2)) * 0.5

        return heatmaps_logp


class NonAutoregressiveDecoder(nn.Module):
    """TODO"""

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
        data: Batch,
        env: Union[str, RL4COEnvBase, None] = None,
        decode_type: str = "multistart_sampling",
        calc_reward: bool = True,
        **strategy_kwargs,
    ):
        # Instantiate environment if needed
        if env is None or isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # calculate heatmap
        heatmaps_logp = self.heatmap_generator(data)

        # setup decoding strategy
        self.decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type, **strategy_kwargs
        )
        td, env, num_starts = self.decode_strategy.pre_decoder_hook(td, env)

        # Main decoding: loop until all sequences are done
        while not td["done"].all():
            log_p, mask = self._get_log_p(td, heatmaps_logp, num_starts)
            td = self.decode_strategy.step(log_p, mask, td)
            td = env.step(td)["next"]

        outputs, actions, td, env = self.decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        return outputs, actions, td

    def _get_log_p(self, td: TensorDict, heatmaps_logp: Tensor, num_starts: int):
        # Get the mask
        mask = ~td["action_mask"]

        current_action = td.get("action", None)
        if current_action is None:
            log_p = heatmaps_logp.mean(-1)
        else:
            batch_size = heatmaps_logp.shape[0]
            _indexer = _multistart_batched_index(batch_size, num_starts)
            log_p = heatmaps_logp[_indexer, current_action, :]

        log_p[mask] = -torch.inf
        log_p = nn.functional.log_softmax(log_p, -1)
        return log_p, mask
