from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase, TransformedEnv, RenameTransform

from ncobench.envs.utils import make_composite_from_td, batch_to_scalar, _set_seed


class TSPEnv(EnvBase):

    batch_locked = False

    def __init__(
        self,
        n_loc: int = 10,
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
            n_loc: number of locations (cities) in the TSP
            td_params: parameters of the environment
            seed: seed for the environment
            device: device to use.  Generally, no need to set as tensors are updated on the fly
        """

        self.n_loc = n_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
   
    @staticmethod
    def get_rewards(loc, actions) -> TensorDict:
        assert (
            torch.arange(actions.size(1), out=actions.data.new()).view(1, -1).expand_as(actions) ==
            actions.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather locations in order of tour
        locs = loc.gather(1, actions.unsqueeze(-1).expand_as(loc))

        # Return the length of the path (L2-norm of difference from each next location from its previous and of last from first)
        locs_next = torch.roll(locs, 1, dims=1)
        return - ((locs_next - locs).norm(p=2, dim=2).sum(1))


    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        prev_a = td["action"]
        first_a = prev_a if batch_to_scalar(td["i"]) == 0 else td["first_a"]
        
        # Set visited to 1
        visited = td["visited"].scatter(
            -1, prev_a[..., None].expand_as(td["visited"]), 1
        )

        # We are done if all the locations have been visited
        done = torch.count_nonzero(visited.squeeze(), dim=-1) >= td["params"]["n_loc"]

        # Calculate reward (minus length of path, since we want to maximize the reward -> minimize the path length)
        # NOTE: reward is calculated outside for now via the get_reward function
        # to calculate here need to pass action sequence or save it as state
        reward = torch.ones_like(done) * (-float("inf"))

        # The output must be written in a ``"next"`` entry
        out = TensorDict(
            {
                "next": {
                    "loc": td["loc"],
                    "first_a": first_a,
                    "prev_a": prev_a,
                    "visited": visited,
                    "i": td["i"] + 1,
                    "params": td["params"],
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )
        return out

    def _reset(self, td: Optional[TensorDict] = None, init_observation = None) -> TensorDict:

        # If no tensordict is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input tensordict contains all the relevant parameters to get started.
        if td is None or td.is_empty():
            bs = self.batch_size if init_observation is None else init_observation.shape[:-2]
            self.device = init_observation.device # set device on the fly
            td = self.gen_params(batch_size=bs)
        batch_size = td.shape  # batch size

        # We do not allow different params (e.g. sizes) on a single batch 
        min_loc = batch_to_scalar(td["params", "min_loc"])
        max_loc = batch_to_scalar(td["params", "max_loc"])
        n_loc = batch_to_scalar(td["params", "n_loc"])

        # We allow loading the initial observation from a dataset for faster loading
        if init_observation is None:
            loc =  torch.rand((*batch_size, n_loc, 2), generator=self.rng) * (max_loc - min_loc) + min_loc
        else:
            loc = init_observation

        # Other variables
        prev_a = torch.zeros((*batch_size, 1), dtype=torch.int64)
        visited = torch.zeros((*batch_size, 1, n_loc), dtype=torch.uint8)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64)

        # Output is a tensordict
        out = TensorDict(
            {
                "loc": loc,
                "first_a": prev_a,
                "prev_a": prev_a,
                "visited": visited,
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
                        "min_loc": self.min_loc,
                        "max_loc": self.max_loc,
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

    def get_mask(self, td: TensorDict) -> torch.Tensor:
        """Returns the mask for the current step"""
        return td['visited'] > 0

    def get_current_node(self, td: TensorDict) -> torch.Tensor:
        return td['prev_a']
    
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
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            # we need to add the "params" to the observation specs, as we want
            # to pass it at each step during a rollout
            params=make_composite_from_td(params),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=n_loc,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rng"] # remove the random number generator for deepcopy piclkling
        return state
    
    def transform(self):
        return TransformedEnv(self, RenameTransform(in_keys=["loc"], out_keys=["observation"], create_copy=True))

    @staticmethod
    def render(td):
        render_tsp(td)

    _set_seed = _set_seed


def render_tsp(td: TensorDict) -> None:

    td = td.detach().cpu()
    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        print("Batch detected. Plotting the first batch element!")
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
    ax.set_title("TSP Solution")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    ax.set_aspect("equal")

    plt.show()