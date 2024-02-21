from math import sqrt
from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)

from rl4co.envs.routing.cvrp import CVRPEnv, CAPACITIES
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)


class CVRPTWEnv(CVRPEnv):
    """
    An implementation of the Capacitated Vehicle Routing Problem (CVRP) with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class.
    """

    name = "cvrptw"

    def __init__(
        self,
        max_loc: int = 150,  # different default value to CVRPEnv to match max_time, will be scaled
        max_time: int = 480,
        scale: bool = False,
        **kwargs,
    ):
        self.min_time = 0  # always 0
        self.max_time = max_time
        self.scale = scale
        super().__init__(max_loc=max_loc, **kwargs)

    def _make_spec(self, td_params: TensorDict):
        super()._make_spec(td_params)

        current_time = UnboundedContinuousTensorSpec(
            shape=(1), dtype=torch.float32, device=self.device
        )

        current_loc = UnboundedContinuousTensorSpec(
            shape=(2), dtype=torch.float32, device=self.device
        )

        durations = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(self.num_loc, 1),
            dtype=torch.int64,
            device=self.device,
        )

        time_windows = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(
                self.num_loc,
                2,
            ),  # each location has a 2D time window (start, end)
            dtype=torch.float32,
            device=self.device,
        )

        # extend observation specs
        self.observation_spec = CompositeSpec(
            **self.observation_spec,
            current_time=current_time,
            current_loc=current_loc,
            durations=durations,
            time_windows=time_windows,
            # vehicle_idx=vehicle_idx,
        )

    def generate_data(self, batch_size) -> TensorDict:
        td = super().generate_data(batch_size)

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        ## define service durations
        # generate randomly (first assume service durations of 0, to be changed later)
        durations = torch.zeros(
            *batch_size, self.num_loc + 1, dtype=torch.float32, device=self.device
        )

        ## define time windows
        # 1. get distances from depot
        dist = get_distance(td["depot"], td["locs"].transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1, device=self.device), dist), dim=1)
        # 2. define upper bound for time windows, lower bound is the distance from the depot
        upper_bound = self.max_time - dist
        # 3. create random values between 0 and 1
        ts_1 = torch.rand(*batch_size, self.num_loc + 1, device=self.device)
        ts_2 = torch.rand(*batch_size, self.num_loc + 1, device=self.device)
        # 4. scale values to lie between their respective min_time and max_time and convert to integer values
        min_ts = (dist + (upper_bound - dist) * ts_1).int()
        max_ts = (dist + (upper_bound - dist) * ts_2).int()
        # 5. set the lower value to min, the higher to max
        min_times = torch.min(min_ts, max_ts)
        max_times = torch.max(min_ts, max_ts)
        # 6. reset times for depot
        min_times[..., :, 0] = 0.0
        max_times[..., :, 0] = self.max_time

        # 7. ensure min_times < max_times to prevent numerical errors in attention.py
        # min_times == max_times may lead to nan values in _inner_mha()
        mask = min_times == max_times
        if torch.any(mask):
            min_tmp = min_times.clone()
            min_tmp[mask] = torch.max(
                dist[mask].int(), min_tmp[mask] - 1
            )  # we are handling integer values, so we can simply substract 1
            min_times = min_tmp

            mask = min_times == max_times  # update mask
            if torch.any(mask):
                max_tmp = max_times.clone()
                max_tmp[mask] = torch.min(
                    torch.floor(upper_bound[mask]).int(),
                    torch.max(
                        torch.ceil(min_tmp[mask] + durations[mask]).int(),
                        max_tmp[mask] + 1,
                    ),
                )
                max_times = max_tmp

        # 8. Adjust durations
        durations = torch.min(durations, max_times - min_times)

        # scale to [0, 1]
        if self.scale:
            durations = durations / self.max_time
            min_times = min_times / self.max_time
            max_times = max_times / self.max_time
            td["depot"] = td["depot"] / self.max_time
            td["locs"] = td["locs"] / self.max_time

        # 9. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # for the case later durations != 0 are used, the durations for the depot must still be 0
        durations[:, 0] = 0.0
        td.update(
            {
                "durations": durations,
                "time_windows": time_windows,
            }
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        not_masked = CVRPEnv.get_action_mask(td)
        batch_size = td["locs"].shape[0]
        current_loc = gather_by_index(td["locs"], td["current_node"]).reshape(
            [batch_size, 2]
        )
        dist = get_distance(current_loc, td["locs"].transpose(0, 1)).transpose(0, 1)
        td.update({"current_loc": current_loc, "distances": dist})
        can_reach_in_time = (
            td["current_time"] + td["durations"] + dist <= td["time_windows"][..., 1]
        )
        return not_masked & can_reach_in_time

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td["locs"].shape[0]
        # update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        td["current_time"] = (td["action"][:, None] != 0) * (
            torch.max(td["current_time"] + distance, start_times) + duration
        )
        # current_node is updated to the selected action
        td = super()._step(td)
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)
        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=self.device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.vehicle_capacity, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, 1, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """The reward is the negative tour length. Time windows
        are not considered for the calculation of the reward."""
        return super().get_reward(td, actions)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        CVRPEnv.check_solution_validity(td, actions)
        batch_size = td["locs"].shape[0]
        # distances to depot
        distances = get_distance(
            td["locs"][..., 0, :], td["locs"].transpose(0, 1)
        ).transpose(0, 1)
        # basic checks on time windows
        assert torch.all(distances >= 0.0), "Distances must be non-negative."
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(
            td["time_windows"][..., :, 1] + distances
            <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
        ), "vehicle cannot get back to depot in time"
        assert torch.all(
            td["durations"] >= 0.0
        ), "Service durations must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] + td["durations"]
            <= td["time_windows"][..., 1] + 1e-6
        ), "service cannot be provided in given time window"
        # check vehicles can meet deadlines
        curr_time = torch.zeros(batch_size, 1, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros_like(curr_time, dtype=torch.int64, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            dist = get_distance(
                gather_by_index(td["locs"], curr_node).reshape([batch_size, 2]),
                gather_by_index(td["locs"], next_node).reshape([batch_size, 2]),
            ).reshape([batch_size, 1])
            curr_time = torch.max(
                curr_time + dist,
                gather_by_index(td["time_windows"], next_node)[..., 0].reshape(
                    [batch_size, 1]
                ),
            ) + gather_by_index(td["durations"], next_node).reshape([batch_size, 1])
            assert torch.all(
                curr_time
                <= gather_by_index(td["time_windows"], next_node)[..., 1].reshape(
                    [batch_size, 1]
                )
            ), "vehicle cannot meet deadline"
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # reset time for depot

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None, scale_xy: bool = False, **kwargs):
        CVRPEnv.render(td=td, actions=actions, ax=ax, scale_xy=scale_xy, **kwargs)

    @staticmethod
    def load_data(
        name: str,
        solomon=False,
        path_instances: str = None,
        type: str = None,
    ):
        if solomon == True:
            assert type in [
                "instance",
                "solution",
            ], "type must be either 'instance' or 'solution'"
            if type == "instance":
                instance = load_solomon_instance(name=name, path=path_instances)
            elif type == "solution":
                instance = load_solomon_solution(name=name, path=path_instances)
            return instance
        return load_npz_to_tensordict(filename=name)

    def extract_from_solomon(self, instance: dict, batch_size: int = 1):
        # convert to format used in CVRPTWEnv
        self.min_demand = instance["demand"][1:].min()
        self.max_demand = instance["demand"][1:].max()
        self.vehicle_capacity = instance["capacity"]
        self.min_loc = instance["node_coord"][1:].min()
        self.max_loc = instance["node_coord"][1:].max()
        self.min_time = instance["time_window"][:, 0].min()
        self.max_time = instance["time_window"][:, 1].max()
        assert self.min_time == 0, "Time window of depot must start at 0."
        assert (
            self.max_time == instance["time_window"][0, 1]
        ), "Depot must have latest end time."
        td = TensorDict(
            {
                "depot": torch.tensor(
                    instance["node_coord"][0],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1),
                "locs": torch.tensor(
                    instance["node_coord"][1:],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1, 1),
                "demand": torch.tensor(
                    instance["demand"][1:],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(batch_size, 1),
                "durations": torch.tensor(
                    instance["service_time"],
                    dtype=torch.int64,
                    device=self.device,
                ).repeat(batch_size, 1),
                "time_windows": torch.tensor(
                    instance["time_window"],
                    dtype=torch.int64,
                    device=self.device,
                ).repeat(batch_size, 1, 1),
            },
            batch_size=1,  # will batch size always be 1 for loaded instances?
        )
        return self.reset(td, batch_size=batch_size)


if __name__ == "__main__":
    import torch
    from rl4co.utils.ops import gather_by_index, get_tour_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instance_name = "R101"

    env = CVRPTWEnv(device=device)
    instance = CVRPTWEnv.load_data(name=instance_name, solomon=True, type="instance")
    solution = CVRPTWEnv.load_data(name=instance_name, solomon=True, type="solution")

    # this sets the environment parameters to the parameters of the solomon instance,
    # in case we'd want to create more instances with the same parameters
    td_test = env.extract_from_solomon(instance, batch_size=1).to(device)

    actions = [0]
    for route in solution["routes"]:
        actions.extend(route)
        actions.append(0)
    actions = torch.tensor(actions, dtype=torch.long, device=env.device).unsqueeze(0)

    # calculate reward without checking validity
    td = td_test.clone()
    # Gather dataset in order of tour
    batch_size = td["locs"].shape[0]
    depot = td["locs"][..., 0:1, :]
    locs_ordered = torch.cat(
        [
            depot,
            gather_by_index(td["locs"], actions).reshape(
                [batch_size, actions.size(-1), 2]
            ),
        ],
        dim=1,
    )
    print("reward:", -get_tour_length(locs_ordered))

    env.get_reward(td_test, actions)
    env.check_solution_validity(td_test, actions)
