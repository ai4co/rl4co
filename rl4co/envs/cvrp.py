from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import gather_by_index

# Default capacities https://arxiv.org/abs/1803.08475
CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}


class CVRPEnv(RL4COEnvBase):
    """
    Capacity Vehicle Routing Problem (CVRP) environment
    At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Args:
        - num_loc <int>: number of locations (cities) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        - min_loc <float>: minimum value for the location coordinates
        - max_loc <float>: maximum value for the location coordinates
        - capacity <float>: capacity of the vehicle
        - td_params <TensorDict>: parameters of the environment
        - seed <int>: seed for the environment
        - device <str>: 'cpu' or 'cuda:0', device to use.  Generally, no need to set as tensors are updated on the fly
    """

    name = "cvrp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_demand: float = 1,
        max_demand: float = 10,
        vehicle_capacity: float = 1.,
        capacity: float = None,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.capacity = CAPACITIES.get(num_loc, None) if capacity is None else capacity
        if self.capacity is None:
            raise ValueError(
                f"Capacity for {num_loc} locations is not defined. Please provide a capacity manually."
            )
        self.vehicle_capacity = vehicle_capacity
        self._make_spec(td_params)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:

        current_node = td["action"][..., None]
        demand = td["demand"]

        # update the used capacity
        demand[..., 0] -= torch.gather(demand, 1, current_node).squeeze()

        # set the visited node demand to 0
        demand.scatter_(-1, current_node, 0)

        # Get the action mask, no zero demand nodes can be visited
        action_mask = torch.abs(demand) > 0

        # Nodes exceeding capacity cannot be visited
        available_capacity = td["capacity"] + demand[..., :1]
        action_mask = torch.logical_and(action_mask, demand <= available_capacity)

        # We are done there are no unvisited locations
        done = torch.count_nonzero(demand, dim=-1) <= 0

        # REVIEW: if all nodes are visited, then set the depot be always available
        action_mask[..., 0] = torch.logical_or(action_mask[..., 0], done)

        # Calculate reward (minus length of path, since we want to maximize the reward -> minimize the path length)
        # Note: reward is calculated outside for now via the get_reward function
        # to calculate here need to pass action sequence or save it as state
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "capacity": td["capacity"],
                    "current_node": current_node,
                    "demand": demand,
                    "action_mask": action_mask,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        # Initialize the current node
        current_node = torch.zeros(
            (*batch_size, 1), dtype=torch.int64, device=self.device
        )

        # Concatenate depot to the locations as the first node
        locs = torch.cat((td["depot"][..., None, :], td["locs"]), dim=-2)

        # Concatenate zero as the first node (depot) to the demand and normalize by the capacity (note that this is not the vehicle capacity)
        demand = torch.cat((torch.zeros_like(td["demand"][..., 0:1]), td["demand"]), dim=-1) / td['capacity'][..., None]

        # Initialize the vehicle capacity
        capacity = torch.full((*batch_size, 1), self.vehicle_capacity, device=self.device)

        # Init the action mask
        action_mask = demand > 0

        return TensorDict(
            {
                "locs": locs,
                "capacity": capacity,
                "current_node": current_node,
                "demand": demand,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

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
                minimum=-self.capacity,
                maximum=self.max_demand,
                shape=(self.num_loc + 1, 1),
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

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        locs = td["locs"]
        # TODO: Check the validation of the tour
        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs = gather_by_index(locs, actions)
        locs_next = torch.roll(locs, 1, dims=1)
        return -((locs_next - locs).norm(p=2, dim=2).sum(1))

    def generate_data(self, batch_size) -> TensorDict:

        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the demand for nodes except the depot
        demand = (
            torch.randint(
                self.min_demand, self.max_demand, size=(*batch_size, self.num_loc)
            )
            .float()
            .to(self.device)
        )

        # Support for heterogeneous capacity if provided
        if not isinstance(self.capacity, torch.Tensor):
            capacity = torch.full((*batch_size,), self.capacity, device=self.device)
        else:
            capacity = self.capacity

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "demand": demand,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")
