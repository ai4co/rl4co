from typing import Any, Generator, NamedTuple, Optional

import torch

from tensordict import TensorDict

from rl4co.utils.ops import gather_by_index


class SHPPMapping(NamedTuple):
    map_action_index: torch.Tensor
    map_node_index: torch.Tensor
    subtsp_coordinates: torch.Tensor


class TSP2SHPPAdapter(object):
    """TODO"""

    def __init__(
        self, parent_td: TensorDict, actions: torch.Tensor, /, min_node_count: int = 20
    ) -> None:
        batch_size = parent_td.batch_size[0]
        n_samples = actions.shape[0] // batch_size
        assert actions.shape[0] == n_samples * batch_size

        self._actions = actions.cpu().clone()
        self.shpp_actions, self.shpp_coordinates = action_partitioner(
            self._actions, parent_td["locs"].cpu(), min_node_count
        )
        # self.shpp_actions and self._actions should share memory

    def get_batched_subtsps(
        self, batch_size: Optional[int] = None
    ) -> Generator[SHPPMapping, Any, None]:
        shpp_count = len(self.shpp_actions)
        if batch_size is None:
            batch_size = shpp_count
        for start_index in range(0, shpp_count, batch_size):
            map_action_index = torch.arange(
                start_index, min(start_index + batch_size, shpp_count)
            )
            map_node_index = self.shpp_actions[map_action_index]  # shpp_index
            shpp_coordinates = self.shpp_coordinates[map_action_index]
            yield SHPPMapping(map_action_index, map_node_index, shpp_coordinates)

    def update_actions(self, mapping: SHPPMapping, subtsp_actions: torch.Tensor):
        self.shpp_actions[mapping.map_action_index] = gather_by_index(
            mapping.map_node_index, subtsp_actions
        )
        # will also modify self._actions

    def get_actions(self):
        return self._actions


def action_partitioner(actions: torch.Tensor, coordinates: torch.Tensor, shpp_nodes: int):
    batch_size = coordinates.shape[0]
    tsp_nodes = actions.shape[1]
    tsp_nodes -= tsp_nodes % shpp_nodes

    shpp_actions = actions[:, :tsp_nodes].view(-1, shpp_nodes)  # trim tail nodes
    repeated_coordinates = (
        coordinates.unsqueeze(1)
        .expand(
            coordinates.shape[0], len(shpp_actions) // batch_size, *coordinates.shape[1:]
        )
        .flatten(0, 1)
    )  # (bs, n_nodes, 2) -> (bs*n_samples*shpp_per_route, n_ndoes, 2)
    shpp_coordinates = gather_by_index(
        repeated_coordinates, shpp_actions
    )  # get shpp coordinates

    return shpp_actions, shpp_coordinates
