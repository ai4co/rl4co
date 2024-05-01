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
from rl4co.utils.ops import gather_by_index, get_distance_matrix, get_tour_length
from rl4co.utils.pylogger import get_pylogger

# For local search
import concurrent.futures
import numpy as np
import numba as nb

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

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Improve the solution using local search, especially 2-opt for TSP.
        Implementation credits to: https://github.com/henry-yeh/DeepACO

        Args:
            td: TensorDict, td from env with shape [batch_size,]
            actions: torch.Tensor, Tour indices with shape [batch_size, num_loc]
            max_iterations: int, maximum number of iterations for 2-opt
            distances: torch.Tensor, distance matrix with shape [batch_size, num_loc, num_loc]
                                     if None, it will be calculated from td["locs"]
        """
        max_iterations = kwargs.get("max_iterations", 1000)

        dists = kwargs.get("distances", None)
        if dists is None:
            dists = get_distance_matrix(td["locs"]).detach().cpu().numpy()
        dists = dists + 1e9 * np.eye(dists.shape[1], dtype=np.float32)[None, :, :]  # fill diagonal with large number

        tours = actions.detach().cpu().numpy().astype(np.uint16)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for dist, tour in zip(dists, tours):
                future = executor.submit(_two_opt_python, distmat=dist, tour=tour, max_iterations=max_iterations)
                futures.append(future)
            return torch.from_numpy(np.stack([f.result() for f in futures]).astype(np.int64)).to(actions.device)

    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        locs = (
            torch.rand((*batch_size, self.num_loc, 2), generator=self.rng)
            * (self.max_loc - self.min_loc)
            + self.min_loc
        )
        return TensorDict({"locs": locs}, batch_size=batch_size)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)


@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i + 1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i - 1], tour[(j + 1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (
                distmat[node_prev, node_j] + distmat[node_i, node_next]
                - distmat[node_prev, node_i] - distmat[node_j, node_next]
            )
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q + 1] = np.flip(tour[p: q + 1])
        return delta
    else:
        return 0.0


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16[:], nb.int64), nogil=True)
def _two_opt_python(distmat, tour, max_iterations=1000):
    iterations = 0
    min_change = -1.0
    while min_change < -1e-6 and iterations < max_iterations:
        min_change = two_opt_once(distmat, tour, 0)
        iterations += 1
    return tour
