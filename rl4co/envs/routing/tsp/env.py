from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import TSPGenerator
from .render import render

log = get_pylogger(__name__)


class TSPEnv(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment.

    Observations:
        - locations of each customer.
        - the current location of the vehicle.

    Constrains:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.

    Finish condition:
        - the agent has visited all customers and returned to the starting customer.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: TSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "tsp"

    def __init__(
        self,
        generator: TSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = TSPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # Visited nodes are set to 0
        action_mask = td["action_mask"].scatter(-1, current_node[:, None], 0)

        # We are done there are no unvisited locations
        done = torch.sum(action_mask, dim=-1) == 0
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "done": done,
                "reward": reward,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        locs = td["locs"]
        num_loc = locs.size(-2)

        first_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        action_mask = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        reward = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)

        td_reset = TensorDict(
            {
                "locs": locs,
                "first_node": first_node,
                "current_node": first_node,
                "i": i,
                "action_mask": action_mask,
                "reward": reward,
            },
            batch_size=batch_size,
        )
        return td_reset

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        locs_ordered = gather_by_index(td["locs"], actions)
        cost = get_tour_length(locs_ordered)
        return -cost

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # Nodes are visited exactly once
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)

    def _make_spec(self, generator: TSPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)
