from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase

from ncobench.envs.utils import make_composite_from_td, batch_to_scalar, _set_seed


class TSPEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(
        self,
        n_loc: int = 10,
        td_params: TensorDict = None,
        seed: int = None,
        device: str = "cpu",
    ):
        """
        Traveling Salesman Problem environment
        At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.
        In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

        Args:
            n_loc: number of locations (cities) in the TSP
            td_params: parameters of the environment
            seed: seed for the environment
            device: device to use.  Generally, no need to set as tensors are updated on the fly
        """

        self.n_loc = n_loc
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        """Take a step in the environment by selecting an action"""

        prev_a = td["action"]  # NOTE was: [:, None]
        ids = td["ids"] if td["ids"][0] >= 0 else None  # compatibility if no batch

        cur_coord = td["loc"][ids, prev_a] if ids is not None else td["loc"][prev_a]

        lengths = td["lengths"]
        if not td["cur_coord"].isnan().all():
            lengths += (cur_coord - td["cur_coord"]).norm(p=2, dim=-1)

        first_a = prev_a if batch_to_scalar(td["i"]) == 0 else td["first_a"]

        # Set visited to 1
        visited = td["visited"].scatter(
            -1, prev_a[..., None].expand_as(td["visited"]), 1
        )

        # We are done if all the locations have been visited
        done = torch.count_nonzero(visited.squeeze(), dim=-1) >= td["params"]["n_loc"]

        # If we are not done, we set the cost to inf (reward is -inf)
        cost = torch.ones_like(done) * float("inf")

        # If we are done, we set the reward to the length of the path, adding the distance to the first location
        if done.any():
            if ids is None:
                cost = lengths + td["dist"][first_a].norm(p=2, dim=-1)
            else:
                cost[done] = (
                    lengths[done] + td["dist"][ids, first_a][done].norm(p=2, dim=-1)
                ).squeeze(-1)

        # The output must be written in a ``"next"`` entry
        out = TensorDict(
            {
                "next": {
                    "loc": td["loc"],
                    "dist": td["dist"],
                    "ids": td["ids"],
                    "first_a": first_a,
                    "prev_a": prev_a,
                    "visited": visited,
                    "lengths": lengths,
                    "cur_coord": cur_coord,
                    "i": td["i"] + 1,
                    "params": td["params"],
                    "reward": -cost,
                    "done": done,
                }
            },
            td.shape,
        )
        return out

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        """Reset the environment"""
        # If no tensordict is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input tensordict contains all the relevant parameters to get started.
        if td is None or td.is_empty():
            td = self.gen_params(batch_size=self.batch_size)
        batch_size = td.shape  # batch size

        # Get unique parameters: we do not allow different params (e.g. sizes) on a single batch
        min_loc = batch_to_scalar(td["params", "min_loc"])
        max_loc = batch_to_scalar(td["params", "max_loc"])
        n_loc = batch_to_scalar(td["params", "n_loc"])

        # Generate locations. We could also load them directy from a dataset
        loc = (
            torch.rand((*batch_size, n_loc, 2), generator=self.rng)
            * (max_loc - min_loc)
            + min_loc
        )

        # Other variables
        dist = (loc[..., :, None, :] - loc[..., None, :, :]).norm(p=2, dim=-1)
        ids = (
            torch.arange(sum([int(dim) for dim in batch_size]), dtype=torch.int64)[
                :, None
            ]
            if len(batch_size) > 0
            else -torch.ones(1, dtype=torch.int64)
        )  # special case for batch_size=()
        prev_a = torch.zeros((*batch_size, 1), dtype=torch.int64)
        visited = torch.zeros((*batch_size, 1, n_loc), dtype=torch.uint8)
        lengths = torch.zeros((*batch_size, 1))
        cur_coord = torch.ones((*batch_size, 1, 2), dtype=torch.float32) * float(
            "nan"
        )  # first coord is nan
        i = torch.zeros(
            (*batch_size, 1), dtype=torch.int64
        )  # Vector with length num_steps

        # Output is a tensordict
        out = TensorDict(
            {
                "loc": loc,
                "dist": dist,
                "ids": ids,
                "first_a": prev_a,
                "prev_a": prev_a,
                "visited": visited,
                "lengths": lengths,
                "cur_coord": cur_coord,
                "i": i,
                "params": td["params"],
            },
            batch_size=batch_size,
        )
        return out

    def gen_params(self, batch_size=None) -> TensorDictBase:
        """Returns a tensordict containing the parameters of the environment"""
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "min_loc": 0,
                        "max_loc": 1,
                        "n_loc": self.n_loc,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    def _make_spec(self, td_params):
        """Make the observation and action specs from the parameters"""
        params = td_params["params"]
        n_loc = params["n_loc"]  # TSP size
        self.observation_spec = CompositeSpec(
            loc=BoundedTensorSpec(
                minimum=params["min_loc"],
                maximum=params["max_loc"],
                shape=(n_loc, 2),
                dtype=torch.float32,
            ),
            dist=UnboundedContinuousTensorSpec(
                shape=(n_loc, n_loc),
                dtype=torch.float32,
            ),
            ids=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            first_a=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            prev_a=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            visited=UnboundedDiscreteTensorSpec(
                shape=(1, n_loc),
                dtype=torch.uint8,
            ),
            lengths=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            cur_coord=UnboundedContinuousTensorSpec(
                shape=(1, 2),
                dtype=torch.float32,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            # we need to add the "params" to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(params),
            shape=(),
        )
        # since the environment is stateless, we expect the previous output as input
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=n_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    @staticmethod
    def plot(td):
        plot_tsp(td)

    _set_seed = _set_seed


def plot_tsp(td: TensorDict) -> None:
    td = td.detach().cpu()
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]

    # Get the coordinates of the visited nodes for the first batch element
    visited_coords = td["loc"][td["visited"][0, 0] == 1][0]

    # Create a plot of the nodes
    fig, ax = plt.subplots()
    ax.scatter(td["loc"][:, 0], td["loc"][:, 1], color="blue")

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
