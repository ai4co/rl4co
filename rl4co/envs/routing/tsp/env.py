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
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from .generator import TSPGenerator
from .render import render

log = get_pylogger(__name__)


class TSPEnv(RL4COEnvBase):
    """Traveling Salesman Problem (TSP) environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

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

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs = td["locs"]

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

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

    def _get_reward(self, td, actions) -> TensorDict:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)

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
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)


class DenseRewardTSPEnv(TSPEnv):
    """
    This is an experimental version of the TSPEnv to be used with stepwise PPO. That is
    this environment defines a stepwise reward function for the TSP which is the distance added
    to the current tour by the given action.
    """

    def __init__(
        self, generator: TSPGenerator = None, generator_params: dict = {}, **kwargs
    ):
        super().__init__(
            generator,
            generator_params,
            check_solution=False,
            _torchrl_mode=True,
            **kwargs,
        )

    def _step(self, td):
        last_node = td["current_node"].clone()
        current_node = td["action"]

        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # calc stepwise reward
        last_node_loc = gather_by_index(td["locs"], last_node)
        curr_node_loc = gather_by_index(td["locs"], current_node)
        reward = get_distance(last_node_loc, curr_node_loc)[:, None]

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _get_reward(self, td, actions=None) -> TensorDict:
        if actions is not None:
            # Gather locations in order of tour and return distance between them (i.e., -reward)
            locs_ordered = gather_by_index(td["locs"], actions)
            return -get_tour_length(locs_ordered)
        return -td["reward"]
