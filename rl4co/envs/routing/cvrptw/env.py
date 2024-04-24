from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)

from .generator import CVRPTWGenerator
from ..cvrp.generator import CVRPGenerator


class CVRPTWEnv(CVRPEnv):
    """Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class in which capacities are considered.
    Additionally considers time windows within which a service has to be started.

    Observations:
        - location of the depot.
        - locations and demand of each customer (city).
        - current location of the vehicle.
        - the remaining capacity of the vehicle.
        - the current time.
        - service durations of each location.
        - time windows of each location.

    Constraints:
        - the tour starts and ends at the depot.
        - each city must be visited exactly once.
        - the vehicle cannot visit cities exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.
        - the vehicle must start the service within the time window of each location.

    Finish Condition:
        - the vehicle has visited all cities and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPTWGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "cvrptw"

    def __init__(
        self,
        generator: CVRPTWGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = CVRPTWGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        """In addition to the calculations in the CVRPEnv, the current time is
        updated to keep track of which nodes are still reachable in time.
        The current_node is updeted in the parent class' _step() function.
        """
        batch_size = td["locs"].shape[0]

        # Update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        td["current_time"] = (td["action"][:, None] != 0) * (
            torch.max(td["current_time"] + distance, start_times) + duration
        )

        # Current_node is updated to the selected action
        td = super()._step(td)
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        """
        not_masked = CVRPEnv.get_action_mask(td)
        batch_size = td["locs"].shape[0]
        current_loc = gather_by_index(td["locs"], td["current_node"]).reshape(
            [batch_size, 2]
        )
        dist = get_distance(current_loc, td["locs"].transpose(0, 1)).transpose(0, 1)
        td.update({"current_loc": current_loc, "distances": dist})
        can_reach_in_time = (
            td["current_time"] + dist <= td["time_windows"][..., 1]
        )  # I only need to start the service before the time window ends, not finish it.
        return not_masked & can_reach_in_time

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        locs = td["locs"]
        num_loc = locs.size(-2)

        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_time = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
        used_capacity = torch.zeros((*batch_size, 1), device=device)
        visited = torch.zeros((*batch_size, num_loc), dtype=torch.bool, device=device)
        done = torch.zeros((*batch_size, 1), dtype=torch.bool, device=device)

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],
                "current_node": current_node,
                "current_time": current_time,
                "used_capacity": used_capacity,
                "capacity": td["capacity"],
                "visited": visited,
                "durations": td["durations"],
                "time_windows": td["time_windows"],
                "done": done,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset


    def _get_reward(self, td: TensorDict, actions: TensorDict) -> torch.Tensor:
        return super()._get_reward(td, actions)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        CVRPEnv.check_solution_validity(td, actions)
        batch_size = td["locs"].shape[0]

        # Distances to depot
        distances = get_distance(
            td["locs"][..., 0, :], td["locs"].transpose(0, 1)
        ).transpose(0, 1)

        # Basic checks on time windows
        assert torch.all(distances >= 0.0), "Distances must be non-negative."
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(
            td["time_windows"][..., :, 0] + distances + td["durations"]
            <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
        ), "vehicle cannot perform service and get back to depot in time."
        assert torch.all(
            td["durations"] >= 0.0
        ), "Service durations must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"

        # Check vehicles can meet deadlines
        curr_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros_like(curr_time, dtype=torch.int64, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            dist = get_distance(
                gather_by_index(td["locs"], curr_node).reshape([batch_size, 2]),
                gather_by_index(td["locs"], next_node).reshape([batch_size, 2]),
            ).reshape([batch_size, 1])
            curr_time = torch.max(
                (curr_time + dist).int(),
                gather_by_index(td["time_windows"], next_node)[..., 0].reshape(
                    [batch_size, 1]
                ),
            )
            assert torch.all(
                curr_time
                <= gather_by_index(td["time_windows"], next_node)[..., 1].reshape(
                    [batch_size, 1]
                )
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["durations"], next_node).reshape(
                [batch_size, 1]
            )
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # Reset time for depot

    @staticmethod
    def load_data(
        name: str,
        solomon=False,
        path_instances: str = None,
        type: str = None,
        compute_edge_weights: bool = False,
    ) -> TensorDict:
        if solomon == True:
            assert type in [
                "instance",
                "solution",
            ], "type must be either 'instance' or 'solution'"
            if type == "instance":
                instance = load_solomon_instance(
                    name=name, path=path_instances, edge_weights=compute_edge_weights
                )
            elif type == "solution":
                instance = load_solomon_solution(name=name, path=path_instances)
            return instance
        return load_npz_to_tensordict(filename=name)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        CVRPEnv.render(td, actions, ax)

    def _make_spec(self, generator: CVRPTWGenerator):
        if isinstance(generator, CVRPGenerator):
            super()._make_spec(generator)
        else:
            current_time = UnboundedContinuousTensorSpec(
                shape=(1), dtype=torch.float32, device=self.device
            )
            current_loc = UnboundedContinuousTensorSpec(
                shape=(2), dtype=torch.float32, device=self.device
            )
            durations = BoundedTensorSpec(
                low=generator.min_time,
                high=generator.max_time,
                shape=(generator.num_loc, 1),
                dtype=torch.int64,
                device=self.device,
            )
            time_windows = BoundedTensorSpec(
                low=generator.min_time,
                high=generator.max_time,
                shape=(
                    generator.num_loc,
                    2,
                ),  # Each location has a 2D time window (start, end)
                dtype=torch.int64,
                device=self.device,
            )
            # Extend observation specs
            self.observation_spec = CompositeSpec(
                **self.observation_spec,
                current_time=current_time,
                current_loc=current_loc,
                durations=durations,
                time_windows=time_windows,
            )
