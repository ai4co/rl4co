import sys; sys.path.append('.')
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from tensordict.tensordict import TensorDict, TensorDictBase
from typing import Optional, Union

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from torchrl.envs import EnvBase, TransformedEnv, RenameTransform

from rl4co.data.dataset import TensorDictDataset
from rl4co.envs.utils import batch_to_scalar, _set_seed, _getstate_env


class CVRPEnv(EnvBase):
    batch_locked = False
    name = "cvrp"

    def __init__(
        self,
        num_loc: int = 10,
        min_loc: float = 0,
        max_loc: float = 1,
        min_demand: float = 0.1,
        max_demand: float = 0.5,
        capacity: float = 1,
        batch_size: list = [],
        seed: int = None,
        device: str = "cpu",
    ):
        """
        Capacity Vehicle Routing Problem (CVRP) environment
        At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.
        In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

        Args:
            - num_loc <int>: number of locations (cities) in the VRP. NOTE: the depot is included
            - min_loc <float>: minimum value for the location coordinates
            - max_loc <float>: maximum value for the location coordinates
            - capacity <float>: capacity of the vehicle
            - td_params <TensorDict>: parameters of the environment
            - seed <int>: seed for the environment
            - device <str>: 'cpu' or 'cuda:0', device to use.  Generally, no need to set as tensors are updated on the fly
        """
        self.device = device
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.capacity = capacity
        self.batch_size = batch_size
        super().__init__(device=device, batch_size=[])

        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        ''' 
        Args:
            - td: <tensor_dict>: tensor dictionary containing the state
            - actions: [batch_size, TODO] num_loc means a sequence of actions till the task is done
        NOTE:
            - about the length of the actions
        '''
        # TODO: Check the validation of the tour

        # Gather dataset in order of tour
        d = td['observation'].gather(1, actions[..., None].expand(*actions.size(), td['observation'].size(-1)))

        # Calculate the reward
        # Length is distance (L3-norm of difference) of each next location to its prev and of first and last to depot
        rewards = (d[..., 1:, :] - d[..., :-1, :]).norm(p=2, dim=-1).sum(-1)

        # Depot to the first node
        rewards += (d[..., :1, :] - td['observation'][..., :1, :]).norm(p=2, dim=-1).sum(-1)

        return rewards

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        ''' Update the states of the environment
        Args:
            - td <TensorDict>: tensor dictionary containing with the action
                - action <int> [batch_size, 1]: action to take
        NOTE:
            - the first node in de demand is larger than 0 or less than 0? 
            - this design is important. For now the design is LESS than 0
        '''
        current_node = td["action"]
        demand = td['demand']

        # update the used capacity
        demand[..., 0] -= torch.gather(demand, 1, current_node).squeeze()

        # set the visited node demand to 0
        demand.scatter_(-1, current_node, 0)

        # Get the action mask, no zero demand nodes can be visited
        action_mask = torch.abs(demand) > 0
        
        # Nodes exceeding capacity cannot be visited
        available_capacity = td['capacity'] + demand[..., :1]
        action_mask = torch.logical_and(action_mask, demand <= available_capacity)

        # We are done there are no unvisited locations
        done = (torch.count_nonzero(demand, dim=-1) <= 0) 

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
                    "observation": td["observation"],
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

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        ''' 
        Args:
            - td (Optional) <TensorDict>: tensor dictionary containing the initial state
        '''
        if batch_size is None:
            batch_size = self.batch_size if td is None else td['observation'].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        # Initialize the current node
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=self.device)

        # Initialize the capacity
        capacity = torch.full((*batch_size, 1), self.capacity)

        # Init the action mask
        action_mask = td['demand'] > 0

        return TensorDict(
            {
                "observation": td["observation"],
                "capacity": capacity,
                "current_node": current_node,
                "demand": td["demand"],
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    def _make_spec(self):
        """ Make the observation and action specs from the parameters. """
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=BoundedTensorSpec(
                minimum=-self.capacity,
                maximum=self.max_demand,
                shape=(self.num_loc, 1),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc, 1),
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

    def generate_data(self, batch_size) -> TensorDict: 
        ''' 
        Args:
            - batch_size <int> or <list>: batch size
        Returns:
            - td <TensorDict>: tensor dictionary containing the initial state
                - observation <Tensor> [batch_size, num_loc, 2]: locations of the nodes
                - demand <Tensor> [batch_size, num_loc]: demand of the nodes
                - capacity <Tensor> [batch_size, 1]: capacity of the vehicle
                - current_node <Tensor> [batch_size, 1]: current node
                - i <Tensor> [batch_size, 1]: number of visited nodes
        NOTE:
            - the observation includes the depot as the first node
            - the demand includes the used capacity at the first value
            - the unvisited variable can be replaced by demand > 0
        '''
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs = torch.FloatTensor(*batch_size, self.num_loc, 2).uniform_(self.min_loc, self.max_loc).to(self.device)

        # Initialize the demand
        demand = torch.FloatTensor(*batch_size, self.num_loc).uniform_(self.min_demand, self.max_demand).to(self.device)

        # The first demand is the used capacity
        demand[..., 0] = 0

        return TensorDict(
            {
                "observation": locs,
                "depot": locs[..., 0, :],
                "demand": demand,
            }, 
            batch_size=batch_size
        )

    def transform(self):
        return self

    @staticmethod
    def render(td):
        render_cvrp(td)

    __getstate__ = _getstate_env

    _set_seed = _set_seed

    def dataset(self, batch_size):
        observation = self.generate_data(batch_size)
        return TensorDictDataset(observation)


def render_cvrp(td):
    # TODO: not finished
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


if __name__ == "__main__":
    # Create a CVRP environment
    env = CVRPEnv(
        num_loc=10, 
        min_loc=0, 
        max_loc=1, 
        min_demand=1, 
        max_demand=10, 
        capacity=100,
        batch_size=[32],
        seed=0,
        device='cpu'
    )

    # REVIEW Test the generate_data()
    td = env.generate_data([64])

    # REVIEW Test the reset()
    env._reset()

    # REVIEW Test the step()
    td['action'] = torch.ones((64, 1), dtype=torch.int64)
    env._step(td)

    # REVIEW Test the reward()
    actions = torch.range(1, 9, dtype=torch.int64).unsqueeze(0).repeat(64, 1)
    actions = torch.cat([actions, torch.zeros((64, 1), dtype=torch.int64)], dim=1)
    print(env.get_reward(td, actions))