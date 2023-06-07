from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index


# Default length capacity
LENGTH_CAPACITY = {20: 2.0, 50: 3.0, 100: 4.0}


class OPEnv(RL4COEnvBase):
    """Orienteering Problem (OP) environment
    At each step, the agent chooses a city to visit. The reward is the -infinite unless the agent visits all the cities.

    Args:
        - num_loc <int>: number of locations (cities) in the VRP. NOTE: the depot is not included
            i.e. for example, if num_loc=20, there would be 21 nodes in total including the depot
        - min_loc <float>: minimum value for the location coordinates
        - max_loc <float>: maximum value for the location coordinates
        - length_capacity <float>: capacity of the vehicle of length, i.e. the maximum length the vehicle can travel
        - td_params <TensorDict>: parameters of the environment
        - seed <int>: seed for the environment
        - device <str>: 'cpu' or 'cuda:0', device to use.  Generally, no need to set as tensors are updated on the fly
    Note:
        - in our setting, the vehicle has to come back to the depot at the end
    """

    name = "op"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_prize: float = 0.1,
        max_prize: float = 0.5,
        length_capacity: float = 1,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_prize = min_prize
        self.max_prize = max_prize
        self.length_capacity = length_capacity
        self._make_spec(td_params)

    def _step(self, td: TensorDict) -> TensorDict:
        """Update the states of the environment
        Args:
            - td <TensorDict>: tensor dictionary containing with the action
                - action <int> [batch_size, 1]: action to take
        NOTE:
            - the first node in de prize is larger than 0 or less than 0?
            - this design is important. For now the design is LESS than 0
        """
        # Update the state
        current_node = td["action"][:, None]

        # Add the length
        current_coord = gather_by_index(td["locs"], current_node, squeeze=False)
        previous_coord = gather_by_index(td["locs"], td["current_node"], squeeze=False)
        used_capacity = td["used_capacity"] + (current_coord - previous_coord).norm(
            p=2, dim=-1
        )

        # Add the collected prize
        selected_prize = gather_by_index(td["prize"], current_node, squeeze=False)
        prize_collect = td["prize_collect"] + selected_prize

        # SECTION: calculate the action mask
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # Get action mask
        exceeds_length = (
            used_capacity + (td["locs"] - current_coord).norm(p=2, dim=-1)
            > td["length_capacity"]
        )
        visited = visited.to(exceeds_length.dtype)
        feasible_actions = ~(visited | visited[..., :1] | exceeds_length[..., None, :])

        # Depot can always be visited
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        feasible_actions[..., 0] = 1

        # We are done if run out the lenght capacity, i.e. no available node to visit
        done = visited.sum(-1) == visited.size(-1)

        # Calculate reward (minus length of path, since we want to maximize the reward -> minimize the path length)
        # Note: reward is calculated outside for now via the get_reward function
        # to calculate here need to pass action sequence or save it as state
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "used_capacity": used_capacity,
                    "length_capacity": td["length_capacity"],
                    "current_node": current_node,
                    "prize": td["prize"],
                    "prize_collect": prize_collect,
                    "visited": visited,
                    "action_mask": feasible_actions,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """
        Args:
            - td (Optional) <TensorDict>: tensor dictionary containing the initial state
        """
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        device = td.device if td is not None else self.device

        # Create loc with depot
        loc_with_depot = torch.cat([td["depot"][..., None, :], td["locs"]], dim=-2)

        # Initialize the current node
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Initialize the capacity
        length_capacity = (
            torch.tensor(LENGTH_CAPACITY[self.num_loc], device=device)
            - (td["depot"][..., None, :] - loc_with_depot).norm(p=2, dim=-1)
            - 1e-6
        )

        # Calculate the lenght of each node back to the depot
        length_to_depot = (loc_with_depot - td["depot"][..., None, :]).norm(p=2, dim=-1)

        used_capacity = torch.zeros((*batch_size, 1)).to(device)

        # Initialize the prize collected
        prize_collect = torch.zeros((*batch_size, 1)).to(device)

        # SECTION: calculate the action mask
        # Visited as mask is easier to understand, as long more memory efficient
        # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
        visited = torch.zeros(
            (*batch_size, 1, self.num_loc + 1), dtype=torch.uint8, device=device
        )

        # Get action mask
        exceeds_length = (
            used_capacity
            + (loc_with_depot - td["depot"][..., None, :]).norm(p=2, dim=-1)
            > length_capacity
        )
        visited = visited.to(exceeds_length.dtype)
        action_mask = visited | visited[..., 0:1] | exceeds_length

        # Depot can always be visited
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        action_mask[:, :, 0] = 0

        return TensorDict(
            {
                "locs": loc_with_depot,
                "prize": td["prize"],
                "used_capacity": used_capacity,
                "length_capacity": length_capacity,
                "length_to_depot": length_to_depot,
                "current_node": current_node,
                "prize_collect": prize_collect,
                "visited": visited,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict = None):
        """Make the observation and action specs from the parameters."""
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
            length_to_depot=BoundedTensorSpec(
                minimum=0,
                maximum=self.length_capacity,
                shape=(1),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
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
        self.reward_spec = UnboundedContinuousTensorSpec()
        self.done_spec = UnboundedDiscreteTensorSpec(dtype=torch.bool)

    def get_reward(self, td, actions) -> TensorDict:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        # Check that tours are valid, i.e. contain 0 to n-1
        sorted_actions = actions.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_actions[:, 1:] == 0)
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"

        # Calculate the reward
        return td["prize_collect"].squeeze(-1)

    def generate_data(self, batch_size, prize_type="dist"):
        """
        Args:
            - batch_size <int> or <list>: batch size
        Returns:
            - td <TensorDict>: tensor dictionary containing the initial state
                - locs <Tensor> [batch_size, num_loc, 2]: locations of the nodes
                - prize <Tensor> [batch_size, num_loc]: prize of the nodes
                - capacity <Tensor> [batch_size, 1]: capacity of the vehicle
        NOTE:
            - the observation includes the depot as the first node
            - the prize includes the used capacity at the first value
            - the unvisited variable can be replaced by prize > 0
        """
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the prize
        if prize_type == "const":
            prize = torch.ones(*batch_size, self.num_loc).to(self.device)
        elif prize_type == "unif":
            prize = (
                1 + torch.randint(0, 100, size=(*batch_size, self.num_loc))
            ) / 100.0
        elif prize_type == "dist":
            dist = (locs_with_depot[..., :1, :] - locs_with_depot[..., 1:, :]).norm(
                p=2, dim=-1
            )
            prize = (
                1 + (dist / dist.max(dim=-1, keepdim=True)[0] * 99).int()
            ).float() / 100.0
        else:
            raise NotImplementedError("Unknown prize type")

        # Depot has no prize
        prize = torch.cat([torch.zeros(*batch_size, 1).to(self.device), prize], dim=-1)

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "prize": prize,
            },
            batch_size=batch_size,
        )

    def render(self, td):
        # TODO
        """Render the environment"""
        raise NotImplementedError
