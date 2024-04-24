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
from rl4co.utils.pylogger import get_pylogger

from .generator import ATSPGenerator
from .render import render

log = get_pylogger(__name__)


class ATSPEnv(RL4COEnvBase):
    """Asymmetric Traveling Salesman Problem (ATSP) environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.
    Unlike the TSP, the distance matrix is asymmetric, i.e., the distance from A to B is not necessarily the same as the distance from B to A.

    Observations:
        - distance matrix between cities
        - the current city
        - the first city (for calculating the reward)
        - the remaining unvisited cities
    
    Constraints:
        - the tour starts and ends at the same city.
        - each city must be visited exactly once.

    Finish Condition:
        - the agent has visited all cities.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: ATSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "atsp"

    def __init__(
        self,
        generator: ATSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = ATSPGenerator(**generator_params)
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
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict], batch_size) -> TensorDict:
        device = td.device
        cost_matrix = td["cost_matrix"]
        num_loc = cost_matrix.size(-2)

        first_node = torch.zeros(batch_size, dtype=torch.int64, device=device)
        i = torch.zeros(batch_size, dtype=torch.int64, device=device)
        action_mask = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        reward = torch.zeros(batch_size, dtype=torch.float32, device=device)

        td_reset = TensorDict(
            {
                "cost_matrix": cost_matrix,
                "first_node": first_node,
                "current_node": first_node,
                "i": i,
                "action_mask": action_mask,
                "reward": reward,
            },
            batch_size=batch_size,
        )
        return td_reset

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> TensorDict:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        distance_matrix = td["cost_matrix"]

        # Get indexes of tour edges
        nodes_src = actions
        nodes_tgt = torch.roll(actions, 1, dims=1)
        batch_idx = torch.arange(
            distance_matrix.shape[0], device=distance_matrix.device
        ).unsqueeze(1)

        return -distance_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)

    def _make_spec(self, generator: ATSPGenerator):
        self.observation_spec = CompositeSpec(
            cost_matrix=BoundedTensorSpec(
                low=generator.min_dist,
                high=generator.max_dist,
                shape=(generator.num_loc, generator.num_loc),
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
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
