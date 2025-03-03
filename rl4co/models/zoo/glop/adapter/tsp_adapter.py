from typing import Any, Generator, NamedTuple, Optional, Union

import torch

from tensordict import TensorDict

from rl4co.utils.ops import gather_by_index


class SHPPMapping(NamedTuple):
    map_action_index: torch.Tensor
    map_node_index: torch.Tensor
    subprob_coordinates: torch.Tensor


class TSP2SHPPAdapter:
    """Adapter class for decomposing and composing TSP solutions."""

    subproblem_env_name = "shpp"

    def __init__(
        self,
        parent_td: TensorDict,
        actions: torch.Tensor,
        subprob_batch_size: Optional[int] = None,
        partition_node_count: Union[int, list[int]] = 20,
        shift: int = 0,
    ) -> None:
        batch_size = parent_td.batch_size[0]
        n_samples = actions.shape[0] // batch_size
        assert actions.shape[0] == n_samples * batch_size

        self._actions = actions.cpu().clone()
        self.shpp_node_counts = (
            [partition_node_count]
            if isinstance(partition_node_count, int)
            else partition_node_count
        )
        self.coordinates = parent_td["locs"].cpu()
        self.subprob_batch_size = subprob_batch_size
        self.shift = shift

    def _get_batched_subprobs_one_iter(
        self, node_count: int
    ) -> Generator[SHPPMapping, Any, None]:
        self.shpp_actions, shpp_coordinates, self.share_memory = self.action_partitioner(
            self._actions, self.coordinates, node_count
        )
        self.shpp_node_count = node_count
        shpp_count = len(self.shpp_actions)
        if shpp_count == 0:
            return
        batch_size = self.subprob_batch_size is None or shpp_count
        for start_index in range(0, shpp_count, batch_size):
            map_action_index = torch.arange(
                start_index, min(start_index + batch_size, shpp_count)
            )
            map_node_index = self.shpp_actions[map_action_index]  # shpp_index
            this_shpp_coordinates = shpp_coordinates[map_action_index]
            yield SHPPMapping(map_action_index, map_node_index, this_shpp_coordinates)

    def get_batched_subprobs(self) -> Generator[SHPPMapping, Any, None]:
        if len(self.shpp_node_counts) == 1:
            yield from self._get_batched_subprobs_one_iter(self.shpp_node_counts[0])
        else:
            for node_count in self.shpp_node_counts:
                yield from self._get_batched_subprobs_one_iter(node_count)
                self.get_actions()  # update self._actions in case not shared
                if self.shift:
                    self._actions = torch.roll(self._actions, self.shift, 1)

    def update_actions(self, mapping: SHPPMapping, subtsp_actions: torch.Tensor):
        self.shpp_actions[mapping.map_action_index] = gather_by_index(
            mapping.map_node_index, subtsp_actions
        )

    def get_actions(self):
        if (
            not self.share_memory and len(self.shpp_actions) > 0
        ):  # write back to actions if memory is not shared
            tsp_nodes = self._actions.shape[1]
            tsp_nodes -= tsp_nodes % self.shpp_node_count
            self._actions[:, :tsp_nodes] = self.shpp_actions.view(-1, tsp_nodes)
        return self._actions

    @staticmethod
    def action_partitioner(
        actions: torch.Tensor, coordinates: torch.Tensor, shpp_nodes: int
    ):
        batch_size, tsp_nodes, _ = coordinates.shape
        share_memory = tsp_nodes % shpp_nodes == 0
        if share_memory:
            shpp_actions = actions.view(-1, shpp_nodes)
        else:
            tsp_nodes -= tsp_nodes % shpp_nodes
            shpp_actions = actions[:, :tsp_nodes].reshape(
                -1, shpp_nodes
            )  # trim tail nodes

        repeated_coordinates = (
            coordinates.unsqueeze(1)
            .expand(
                coordinates.shape[0],
                len(shpp_actions) // batch_size,
                *coordinates.shape[1:],
            )
            .flatten(0, 1)
        )  # (bs, n_nodes, 2) -> (bs*n_samples*shpp_per_route, n_nodes, 2)
        shpp_coordinates = gather_by_index(repeated_coordinates, shpp_actions)

        return shpp_actions, shpp_coordinates, share_memory
