from typing import Optional
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from zmq import device

from rl4co.envs.routing.cvrp import CVRPEnv, CAPACITIES
from rl4co.utils.ops import gather_by_index, get_distance


class CVRPTWEnv(CVRPEnv):
    """
    An implementation of the Capacitated Vehicle Routing Problem (CVRP) with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class.
    """

    name = "cvrptw"

    def __init__(
        self,
        min_time: float = 0.0,
        max_time: float = 100.0,
        **kwargs,
    ):
        self.min_time = min_time
        self.max_time = max_time
        super().__init__(**kwargs)

    def _make_spec(self, td_params: TensorDict):
        super()._make_spec(td_params)

        current_time = UnboundedContinuousTensorSpec(shape=(1), dtype=torch.float32)

        durations = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(self.num_loc, 1),
            dtype=torch.float32,
        )

        time_windows = BoundedTensorSpec(
            low=self.min_time,
            high=self.max_time,
            shape=(
                self.num_loc,
                2,
            ),  # each location has a 2D time window (start, end)
            dtype=torch.float32,
        )

        num_vehicles = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.int64)
        # vehicle_idx = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.int64)

        # extend observation specs
        self.observation_spec = CompositeSpec(
            **self.observation_spec,
            current_time=current_time,
            durations=durations,
            num_vehicles=num_vehicles,
            time_windows=time_windows,
            # vehicle_idx=vehicle_idx,
        )

    def generate_data(self, batch_size) -> TensorDict:
        td = super().generate_data(batch_size)
        # batch_size = [td["locs"].shape[0]]
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # initialize at zero
        current_time = torch.zeros(*batch_size, 1, dtype=torch.float32)
        num_vehicles = torch.ones(*batch_size, dtype=torch.int64)

        # define time windows
        min_ts = torch.FloatTensor(*batch_size, self.num_loc + 1).uniform_(
            self.min_time, self.max_time
        )
        max_ts = torch.FloatTensor(*batch_size, self.num_loc + 1).uniform_(
            self.min_time, self.max_time
        )
        min_times, max_times = torch.min(min_ts, max_ts), torch.max(min_ts, max_ts)
        time_windows = torch.stack((min_times, max_times), dim=-1)
        time_windows[:, 0, :] = 0.0  # depot has no time window

        # first assume service durations of 0 (to be changed later)
        durations = torch.zeros(*batch_size, self.num_loc + 1, dtype=torch.float32)
        # for the case later that durations != 0 are used, the durations for the depot must still be 0
        durations[:, 0] = 0.0
        td.update(
            {
                "current_time": current_time,
                "durations": durations,
                "num_vehicles": num_vehicles,
                "time_windows": time_windows,
            }
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        masked = CVRPEnv.get_action_mask(td)
        current_loc = gather_by_index(td["locs"], td["current_node"])
        # mask_locs = torch.arange(td["locs"].size(1))[None, :] != td["current_node"]
        # remaining_locs = (
        #     td["locs"]
        #     .masked_select(mask_locs.unsqueeze(-1))
        #     .reshape([3, td["locs"].size(1) - 1, 2])
        # )
        dist = get_distance(current_loc, td["locs"].transpose(0, 1)).transpose(0, 1)
        can_reach_in_time = (
            td["current_time"] + td["durations"] + dist
            # + dist[mask_locs].reshape(td["locs"].size(0), td["locs"].size(1) - 1)
            <= td["time_windows"][:, :, 1]
        )
        # TODO include logic to send vehicles back to the depot
        # when their capacity is full but other customers still need to be served
        # in which case we need to mask all nodes except the depot and reset current_time to 0
        return ~masked & can_reach_in_time

    # def _step(self, td: TensorDict) -> TensorDict:
    #     return super()._step(td)

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
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
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
                "current_time": td["current_time"],
                "durations": td["durations"],
                "num_vehicles": td["num_vehicles"],
                "time_windows": td["time_windows"],
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
