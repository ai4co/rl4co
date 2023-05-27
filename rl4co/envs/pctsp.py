from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs import RL4COEnvBase


class PCTSPEnv(RL4COEnvBase):
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

    name = "pctsp"

    def __init__(
        self,
        num_loc: int = 10,
        min_loc: float = 0,
        max_loc: float = 1,
        min_prize: float = 0.1,
        max_prize: float = 0.5,
        min_penalty: float = 0.1,
        max_penalty: float = 0.5,
        require_prize: float = 1,
        batch_size: list = [],
        td_params: TensorDict = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.require_prize = require_prize
        self.batch_size = batch_size

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        """Update the states of the environment
        Args:
            - td <TensorDict>: tensor dictionary containing with the action
                - action <int> [batch_size, 1]: action to take
        NOTE:
            - the first node in de demand is larger than 0 or less than 0?
            - this design is important. For now the design is LESS than 0
        """
        current_node = td["action"][..., None]
        prize = td["prize"]
        prize_collect = td["prize_collect"]

        # Collect the prize
        prize_collect += torch.gather(td["prize"], 1, current_node)

        # Update the prize
        prize.scatter_(-1, current_node, 0)

        # Get the action mask, no zero demand nodes can be visited
        action_mask = prize > 0

        # If collected prize is larger than required prize, then the depot is allowed to visit
        action_mask[..., :1] = torch.logical_or(
            action_mask[..., :1], prize_collect >= td["prize_require"]
        )

        # Force to done when there are no unvisited locations
        done = (torch.count_nonzero(prize, dim=-1) <= 0)[..., None]

        # We can choose to finish when we meet the required prize
        # The mark of finish is back to the depot
        done = torch.logical_or(done, current_node == 0)

        # If done, then set the depot be always available
        action_mask[..., :1] = torch.logical_or(action_mask[..., :1], done)

        # If done, then we are not allowed to visit any other nodes
        action_mask[..., 1:] = torch.logical_xor(
            torch.logical_or(action_mask[..., 1:], done), done
        )

        # Calculate reward (minus length of path, since we want to maximize the reward -> minimize the path length)
        # Note: reward is calculated outside for now via the get_reward function
        # to calculate here need to pass action sequence or save it as state
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "observation": td["observation"],
                    "current_node": current_node,
                    "prize": prize,
                    "prize_collect": prize_collect,
                    "prize_require": td["prize_require"],
                    "penalty": td["penalty"],
                    "action_mask": action_mask,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        """
        Args:
            - td (Optional) <TensorDict>: tensor dictionary containing the initial state
        """
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["observation"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        # Initialize the current node
        current_node = torch.zeros(
            (*batch_size, 1), dtype=torch.int64, device=self.device
        )

        # Collected prize
        prize_collect = torch.zeros(
            (*batch_size, 1), dtype=torch.float32, device=self.device
        )

        # Required prize
        prize_require = torch.full(
            (*batch_size, 1),
            self.require_prize,
            dtype=torch.float32,
            device=self.device,
        )

        # Init the action mask
        action_mask = td["prize"] > 0

        return TensorDict(
            {
                "observation": td["observation"],
                "current_node": current_node,
                "prize": td["prize"],
                "prize_collect": prize_collect,
                "prize_require": prize_require,
                "penalty": td["penalty"],
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
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
            prize=BoundedTensorSpec(
                minimum=self.min_prize,
                maximum=self.max_prize,
                shape=(self.num_loc),
                dtype=torch.float32,
            ),
            prize_collect=UnboundedContinuousTensorSpec(
                shape=(1,),
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

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        # Calculate the length
        locs = td["observation"]
        locs = locs.gather(1, actions[..., None].expand(*actions.size(), locs.size(-1)))
        locs_next = torch.roll(locs, 1, dims=1)
        length = -((locs_next - locs).norm(p=2, dim=2).sum(1))

        # Calculate the penalty
        penalty = torch.sum(td["penalty"] * (td["prize"] > 0).float(), dim=-1)
        return length + penalty

    def generate_data(self, batch_size) -> TensorDict:
        """
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
        """
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs = (
            torch.FloatTensor(*batch_size, self.num_loc, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the prize
        prize = (
            torch.FloatTensor(*batch_size, self.num_loc)
            .uniform_(self.min_prize, self.max_prize)
            .to(self.device)
        )

        # Initialize the penalty
        penalty = (
            torch.FloatTensor(*batch_size, self.num_loc)
            .uniform_(self.min_penalty, self.max_penalty)
            .to(self.device)
        )

        # The depot prize and penalty are zero
        prize[..., 0] = 0
        penalty[..., 0] = 0

        return TensorDict(
            {
                "observation": locs,
                "depot": locs[..., 0, :],
                "prize": prize,
                "penalty": penalty,
            },
            batch_size=batch_size,
        )

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")
