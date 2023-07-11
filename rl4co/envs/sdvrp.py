from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.cvrp import CVRPEnv
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SDVRPEnv(CVRPEnv):
    """Split Delivery Vehicle Routing Problem (SDVRP) environment.
    SDVRP is a generalization of CVRP, where nodes can be visited multiple times and a fraction of the demand can be met.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is the -infinite unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Args:
        num_loc: number of locations (cities) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        min_demand: minimum value for the demand of each customer
        max_demand: maximum value for the demand of each customer
        vehicle_capacity: capacity of the vehicle
        capacity: capacity of the vehicle
        td_params: parameters of the environment
    """

    name = "sdvrp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_demand: float = 1,
        max_demand: float = 10,
        vehicle_capacity: float = 1.0,
        capacity: float = None,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            min_demand=min_demand,
            max_demand=max_demand,
            vehicle_capacity=vehicle_capacity,
            capacity=capacity,
            td_params=td_params,
            **kwargs,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        # Update the state
        current_node = td["action"][:, None]  # Add dimension for step

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand_with_depot"], current_node, dim=-1, squeeze=False
        )[..., :1]
        delivered_demand = torch.min(
            selected_demand, td["vehicle_capacity"] - td["used_capacity"]
        )

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + delivered_demand) * (
            current_node != 0
        ).float()

        # Update demand
        demand_with_depot = td["demand_with_depot"].scatter_add(
            -1, current_node, -delivered_demand
        )

        # Get done and reward (-inf since we get it outside)
        done = ~(demand_with_depot > 0).any(-1)
        reward = torch.ones_like(done) * float("-inf")

        td_step = TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "demand": td["demand"],
                    "demand_with_depot": demand_with_depot,
                    "current_node": current_node,
                    "used_capacity": used_capacity,
                    "vehicle_capacity": td["vehicle_capacity"],
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )
        td_step["next"].set("action_mask", self.get_action_mask(td_step["next"]))
        return td_step

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.device = td["locs"].device

        # Create reset TensorDict
        reset_td = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "demand_with_depot": torch.cat(
                    (torch.zeros_like(td["demand"][..., 0:1]), td["demand"]), -1
                ),
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=self.device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.vehicle_capacity, device=self.device
                ),
            },
            batch_size=batch_size,
        )
        reset_td.set("action_mask", self.get_action_mask(reset_td))
        return reset_td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        mask_loc = (td["demand_with_depot"][..., 1:] == 0) | (
            td["used_capacity"] >= td["vehicle_capacity"]
        )
        mask_depot = (td["current_node"] == 0).squeeze(-1) & (
            (mask_loc == 0).int().sum(-1) > 0
        )
        return ~torch.cat((mask_depot[..., None], mask_loc), -1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that the solution is valid (all demand is satisfied)"""

        batch_size, graph_size = td["demand"].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat((-td["vehicle_capacity"], td["demand"]), 1)

        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(td["demand"][..., 0])
        a_prev = None
        for a in actions.transpose(0, 1):
            assert (
                a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all()
            ), "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], td["vehicle_capacity"].squeeze(-1) - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=BoundedTensorSpec(
                minimum=self.min_demand,
                maximum=self.max_demand,
                shape=(self.num_loc, 1),  # demand is only for customers
                dtype=torch.float32,
            ),
            demand_with_depot=BoundedTensorSpec(
                minimum=self.min_demand,
                maximum=self.max_demand,
                shape=(self.num_loc + 1, 1),
                dtype=torch.float32,
            ),
            used_capacity=BoundedTensorSpec(
                minimum=0,
                maximum=self.vehicle_capacity,
                shape=(1,),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
