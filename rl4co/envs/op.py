from collections import defaultdict
from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.data.dataset import TensorDictDataset


class OPEnv(EnvBase):
    batch_locked = False

    def __init__(
        self,
        num_loc: int = 10,
        min_loc: float = 0,
        max_loc: float = 1,
        min_demand: float = 0.1,
        max_demand: float = 0.5,
        length_capacity: float = 1,
        batch_size: list = [],
        td_params: TensorDict = None,
        seed: int = None,
        device: str = "cpu",
        **kwargs,
    ):
        """ Orienteering Problem (OP) environment
        At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.

        Args:
            - num_loc <int>: number of locations (cities) in the VRP. NOTE: the depot is included
            - min_loc <float>: minimum value for the location coordinates
            - max_loc <float>: maximum value for the location coordinates
            - length_capacity <float>: capacity of the vehicle of length, i.e. the maximum length the vehicle can travel
            - td_params <TensorDict>: parameters of the environment
            - seed <int>: seed for the environment
            - device <str>: 'cpu' or 'cuda:0', device to use.  Generally, no need to set as tensors are updated on the fly
        """
        super().__init__(device=device, batch_size=[])
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.length_capacity = length_capacity
        self.batch_size = batch_size
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _step(self, td: TensorDict) -> TensorDict:
        ''' Update the states of the environment
        Args:
            - td <TensorDict>: tensor dictionary containing with the action
                - action <int> [batch_size, 1]: action to take
        NOTE:
            - the first node in de demand is larger than 0 or less than 0? 
            - this design is important. For now the design is LESS than 0
        '''
        current_node = td["action"]
        length_capacity = td["length_capacity"]
        prize = td['prize']
        prize_collect = td['prize_collect']

        # Collect prize
        prize_collect += torch.gather(prize, 1, current_node).squeeze()

        # Set the visited node demand to -1
        prize.scatter_(-1, current_node, -1)

        # Update the used length capacity
        length_capacity -= (
            torch.gather(td["observation"], -2, current_node) - 
            torch.gather(td["observation"], -2, td["previous_node"])
            ).norm(p=2, dim=-1)

        # Get the action mask, no zero demand nodes can be visited
        action_mask = torch.abs(prize) >= 0
        
        # Nodes distance exceeding length capacity cannot be visited
        length_to_next_node = (
            td["observation"] - torch.gather(td["observation"], -2, current_node)
            ).norm(p=2, dim=-1)
        action_mask = torch.logical_and(action_mask, length_to_next_node <= length_capacity)

        # We are done if run out the lenght capacity, i.e. no available node to visit
        done = (torch.count_nonzero(action_mask.float(), dim=-1) <= 0) 

        # Calculate reward (minus length of path, since we want to maximize the reward -> minimize the path length)
        # Note: reward is calculated outside for now via the get_reward function
        # to calculate here need to pass action sequence or save it as state
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "observation": td["observation"],
                    "length_capacity": td["length_capacity"],
                    "previsou_node": current_node,
                    "prize": prize,
                    "prize_collect": prize_collect,
                    "action_mask": action_mask,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )


    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
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
        length_capacity = torch.full((*batch_size, 1), self.length_capacity)

        # Init the action mask
        action_mask = td['demand'] > 0

        return TensorDict(
            {
                "observation": td["observation"],
                "length_capacity": length_capacity,
                "previous_node": current_node,
                "prize": td["prize"],
                "prize_collect": torch.zeros_like(td["prize"]),
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )


    def _make_spec(self, td_params: TensorDict = None):
        """ Make the observation and action specs from the parameters. """
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            length_capacity=BoundedTensorSpec(
                minimum=0,
                maximum=self.length_capacity,
                shape=(1),
                dtype=torch.float32,
            ),
            previous_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            prize=BoundedTensorSpec(
                minimum=-1,
                maximum=self.max_prize,
                shape=(self.num_loc),
                dtype=torch.float32,
            ),
            prize_collect=UnboundedContinuousTensorSpec(
                shape=(self.num_loc),
                dtype=torch.float32,
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
    
    def get_reward(self, td, actions) -> TensorDict:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        raise td["prize_collect"]
    
    def dataset(self, batch_size):
        """Return a dataset of observations"""
        observation = self.generate_data(batch_size)
        return TensorDictDataset(observation)

    def generate_data(self, batch_size):
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
        prize = torch.FloatTensor(*batch_size, self.num_loc).uniform_(self.min_prize, self.max_prize).to(self.device)

        # The first demand is the used capacity
        prize[..., 0] = 0

        return TensorDict(
            {
                "observation": locs,
                "depot": locs[..., 0, :],
                "prize": prize,
            }, 
            batch_size=batch_size
        )

    def transform(self):
        """Used for converting TensorDict variables (such as with torch.cat) efficiently
        https://pytorch.org/rl/reference/generated/torchrl.envs.transforms.Transform.html
        """
        return self

    def render(self, td):
        """Render the environment"""
        raise NotImplementedError

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator as it is not allowed by deepcopy
        """
        state = self.__dict__.copy()
        del state["rng"]
        return state

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng