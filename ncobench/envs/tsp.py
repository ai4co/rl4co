from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase, TransformedEnv, RenameTransform

from ncobench.data.dataset import TensorDictDataset
from ncobench.envs.utils import batch_to_scalar, _set_seed, _getstate_env


class TSPEnv(EnvBase):
    batch_locked = False
    name = "tsp"

    def __init__(
        self,
        num_loc: int = 10,
        min_loc: float = 0,
        max_loc: float = 1,
        td_params: TensorDict = None,
        seed: int = None,
        device: str = "cpu",
    ):
        """
        Traveling Salesman Problem environment
        At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.
        In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

        Args:
            num_loc: number of locations (cities) in the TSP
            td_params: parameters of the environment
            seed: seed for the environment
            device: device to use.  Generally, no need to set as tensors are updated on the fly
        """

        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc

        super().__init__(device=device, batch_size=[])
        # self._make_spec(td_params)
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        loc = td["observation"]
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather locations in order of tour
        locs = loc.gather(1, actions.unsqueeze(-1).expand_as(loc))

        # Return the length of the path (L2-norm of difference from each next location from its previous and of last from first)
        locs_next = torch.roll(locs, 1, dims=1)
        return -((locs_next - locs).norm(p=2, dim=2).sum(1))

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node[..., None].expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = (
            torch.count_nonzero(available.squeeze(), dim=-1) <= 0
        )  # td["params"]["num_loc"]

        # Calculate reward (minus length of path, since we want to maximize the reward -> minimize the path length)
        # NOTE: reward is calculated outside for now via the get_reward function
        # to calculate here need to pass action sequence or save it as state
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "observation": td["observation"],
                    "first_node": first_node,
                    "current_node": current_node,
                    "i": td["i"] + 1,
                    "action_mask": available,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(
        self, td: Optional[TensorDict] = None, init_obs=None, batch_size=None
    ) -> TensorDict:
        # If no tensordict (or observations tensor) is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input tensordict contains all the relevant parameters to get started.
        init_locs = td["observation"] if td is not None else init_obs
        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        device = init_locs.device if init_locs is not None else self.device
        self.device = device

        # We allow loading the initial observation from a dataset for faster loading
        if init_locs is None:
            # number generator is on CPU by default, set device after
            init_locs = self.generate_data(batch_size=batch_size).to(device)

        # Other variables
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, 1, self.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "observation": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self):
        """Make the observation and action specs from the parameters"""
        # params = td_params["params"]
        # num_loc = params["num_loc"]  # TSP size
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                # minimum=params["min_loc"],
                # maximum=params["max_loc"],
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
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
                shape=(1, self.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def dataset(self, batch_size):
        observation = self.generate_data(batch_size)
        return TensorDictDataset(observation)
    
    def generate_data(self, batch_size):
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        locs = (
            torch.rand((*batch_size, self.num_loc, 2), generator=self.rng)
            * (self.max_loc - self.min_loc)
            + self.min_loc
        )
        return locs

    def transform(self):
        return self

    __getstate__ = _getstate_env

    _set_seed = _set_seed

    @staticmethod
    def render_tsp(td):
        import matplotlib.pyplot as plt

        td = td.detach().cpu()
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]

        key = "observation" if "observation" in td.keys() else "loc"

        # Get the coordinates of the visited nodes for the first batch element
        visited_coords = td[key][td["action_mask"][0, 0] == 0][0]

        # Create a plot of the nodes
        fig, ax = plt.subplots()
        ax.scatter(td[key][:, 0], td[key][:, 1], color="blue")

        # Plot the visited nodes
        ax.scatter(visited_coords[:, 0], visited_coords[:, 1], color="red")

        # Add arrows between visited nodes as a quiver plot
        x = visited_coords[:, 0]
        y = visited_coords[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)

        # Colors via a colormap
        cmap = plt.get_cmap("cividis")
        norm = plt.Normalize(vmin=0, vmax=len(x))
        colors = cmap(norm(range(len(x))))

        ax.quiver(
            x[:-1], y[:-1], dx, dy, scale_units="xy", angles="xy", scale=1, color=colors
        )

        # Add final arrow from last node to first node
        ax.quiver(
            x[-1],
            y[-1],
            x[0] - x[-1],
            y[0] - y[-1],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="red",
            linestyle="dashed",
        )

        # Plot numbers inside circles next to visited nodes
        for i, coord in enumerate(visited_coords):
            ax.add_artist(plt.Circle(coord, radius=0.02, color=colors[i]))
            ax.annotate(
                str(i + 1), xy=coord, fontsize=10, color="white", va="center", ha="center"
            )

        # Set plot title and axis labels
        ax.set_title("TSP Solution\nTotal length: {:.2f}".format(-td["reward"][0]))
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_aspect("equal")

        plt.show()
