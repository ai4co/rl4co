from typing import Optional

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.utils import batch_to_scalar
from rl4co.envs.base import RL4COEnvBase


class TSPEnv(RL4COEnvBase):
    name = "tsp"

    def __init__(
        self,
        num_loc: int = 20,
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
        super().__init__(seed=seed, device=device)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self._make_spec(td_params)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) <= 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "locs": td["locs"],
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

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        init_locs = td["locs"] if td is not None else None
        if batch_size is None:
            batch_size = self.batch_size if init_locs is None else init_locs.shape[:-2]
        self.device = device = (
            init_locs.device if init_locs is not None else self.device
        )
        if init_locs is None:
            init_locs = self.generate_data(batch_size=batch_size).to(device)[
                "locs"
            ]

        # Other variables
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params):
        """Make the observation and action specs from the parameters"""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
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
                shape=(self.num_loc),
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

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        locs = td["locs"]
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs = locs.gather(1, actions.unsqueeze(-1).expand_as(locs))
        locs_next = torch.roll(locs, 1, dims=1)
        return -((locs_next - locs).norm(p=2, dim=2).sum(1))

    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        locs = (
            torch.rand((*batch_size, self.num_loc, 2), generator=self.rng)
            * (self.max_loc - self.min_loc)
            + self.min_loc
        )
        return TensorDict({"locs": locs}, batch_size=batch_size)

    @staticmethod
    def render(td):
        import matplotlib.pyplot as plt

        td = td.detach().cpu()
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]

        key = "locs" if "locs" in td.keys() else "loc"

        locs = td[key]
        x = locs[:, 0]
        y = locs[:, 1]

        # Create a plot of the nodes
        fig, ax = plt.subplots()
        ax.scatter(td[key][:, 0], td[key][:, 1], color="blue")

        # Plot the visited nodes
        ax.scatter(x, y, color="red")

        # Add arrows between visited nodes as a quiver plot
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
        for i, coord in enumerate(locs):
            ax.add_artist(plt.Circle(coord, radius=0.02, color=colors[i]))
            ax.annotate(
                str(i + 1),
                xy=coord,
                fontsize=10,
                color="white",
                va="center",
                ha="center",
            )

        # Set plot title and axis labels
        ax.set_title(
            "TSP Solution\nTotal length: {:.2f}".format(-td["reward"].detach().item())
        )
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_aspect("equal")
        plt.show()
