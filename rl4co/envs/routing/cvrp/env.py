from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import CVRPGenerator
from .render import render

log = get_pylogger(__name__)


class CVRPEnv(RL4COEnvBase):
    """Capacitated customers Routing Problem (CVRP) environment.
    
    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle,

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "cvrp"

    def __init__(
        self,
        generator: CVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = CVRPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]

        # Get demand for selected node
        selected_demand = gather_by_index(td["demand"], current_node, squeeze=False)

        # If not depot, add demand to used capacity; if depot, reset used capacity to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (current_node != 0).float()

        # Update visited nodes
        visited = td["visited"].scatter(-1, current_node, True)

        done = visited.sum(-1, keepdim=True) == visited.size(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        action_mask = ~td["visited"]

        # Visiting node demand can not exceed vehicle remaining capacity
        within_capacity_flag = (
            td["demand"] + td["used_capacity"] < td["capacity"]
        )
        action_mask &= within_capacity_flag

        # Depot is always available except the vehicle is already at the depot while there are still unserved nodes
        action_mask[:, :1] = (td["current_node"] != 0) | td["done"]

        return action_mask

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        locs = td["locs"]
        num_loc = locs.size(-2)
        
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        used_capacity = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
        visited = torch.zeros((*batch_size, num_loc), dtype=torch.bool, device=device)
        done = torch.zeros((*batch_size, 1), dtype=torch.bool, device=device)

        # Depot is always visited
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],
                "current_node": current_node,
                "used_capacity": used_capacity,
                "capacity": td["capacity"],
                "visited": visited,
                "done": done,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> torch.Tensor:
        # Gather dataset in order of tour
        batch_size = td["locs"].shape[0]
        depot = td["locs"][..., :1, :]
        locs_ordered = torch.cat(
            [
                depot,
                gather_by_index(td["locs"], actions).reshape(
                    [batch_size, actions.size(-1), 2]
                ),
            ],
            dim=1,
        )
        cost = get_tour_length(locs_ordered)
        return -cost

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        _, num_loc, _ = td["locs"].size()
        sorted_pi = actions.data.sort(1)[0]

        # Nodes are visited exactly once
        assert (
            torch.arange(1, num_loc, out=sorted_pi.data.new()).view(1, -1)
            == sorted_pi[:, -num_loc+1:]
        ).all() and (sorted_pi[:, :-num_loc+1] == 0).all(), "Invalid tour"

        # Used capacity does not exceed vehicle capacity
        action_demand = td["demand"].gather(1, actions)
        used_capacity = torch.zeros(td.batch_size).to(td.device)
        for i in range(actions.size(1)):
            used_capacity += action_demand[:, i] 
            used_capacity[actions[:, i] == 0] = 0 # Reset capacity at depot
            assert (
                used_capacity <= td["capacity"] + 1e-5
            ).all(), "Used more than capacity"

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        td_load = load_npz_to_tensordict(fpath)
        td_load.set("demand", td_load["demand"] / td_load["unnorm_capacity"][:, None])
        return td_load

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)

    def _make_spec(self, generator: CVRPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=BoundedTensorSpec(
                low=-generator.capacity,
                high=generator.max_demand,
                shape=(generator.num_loc + 1, 1),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
