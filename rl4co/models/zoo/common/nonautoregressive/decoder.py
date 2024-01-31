from typing import Callable, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.dec_strategies import DecodingStrategy, get_decoding_strategy
from rl4co.utils.ops import select_start_nodes
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class NonAutoregressiveDecoder(nn.Module):
    """TODO"""

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int,
        num_layers: int,
        select_start_nodes_fn: Callable = select_start_nodes,
        act_fn: Union[str, Callable] = "silu",
        linear_bias: bool = True,
    ) -> None:
        super(NonAutoregressiveDecoder, self).__init__()

        self.env_name = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        self.linears = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim, bias=linear_bias)
                for _ in range(num_layers)
            ]
        )
        self.output = nn.Linear(embedding_dim, 1, bias=linear_bias)
        self.activation = (
            getattr(nn.functional, act_fn) if isinstance(act_fn, str) else act_fn
        )

        self.select_start_nodes_fn = select_start_nodes_fn

    def forward(
        self,
        td: TensorDict,
        data: Batch,
        env: Union[str, RL4COEnvBase, None] = None,
        decode_type: str = "sampling",
        calc_reward: bool = True,
        **strategy_kwargs,
    ):
        # Instantiate environment if needed
        if isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # calculate heatmap
        # batch_logits = self._mlp_decode(data)
        # heatmaps_logp = self._make_heatmaps(batch_logits)

        # setup decoding strategy
        self.decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type, **strategy_kwargs
        )

        # Main decoding: loop until all sequences are done
        # while not td["done"].all():
        #     log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp, num_starts)
        #     td = self.decode_strategy.step(log_p, mask, td)
        #     td = env.step(td)["next"]

    def _mlp_decode(self, data):
        edge_attr = data.edge_attr
        for layer in self.linears:
            edge_attr = self.activation(layer(edge_attr))
        data.edge_attr = self.output(edge_attr)
        return data

    def _make_heatmaps(self, batch_graph: Batch) -> Tensor:
        graphs = batch_graph.to_data_list()
        device = graphs[0].edge_attr.device
        batch_size = len(graphs)
        num_nodes = graphs[0].shape[0]

        heatmaps_logp = (
            torch.zeros((batch_size, num_nodes, num_nodes), device=device) - 1e9
        )
        for index, graph in enumerate(graphs):
            edge_index, edge_attr = graph.edge_index, graph.edge_attr
            heatmaps_logp[index, edge_index[0], edge_index[1]] = edge_attr

        return heatmaps_logp
