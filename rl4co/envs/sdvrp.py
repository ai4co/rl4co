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

from rl4co.envs.base import RL4COEnvBase


class SDVRPEnv(RL4COEnvBase):
    name = "sdvrp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        capacity: float = 1.,
        td_params: TensorDict = None,
        seed: int = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(seed=seed, device=device)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.capacity = capacity
        self._make_spec(td_params)

    def _step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        Gives the next observation, reward, done
        """
        current_node = td['action']
        current_coord = torch.gather(td['observation'], 1, current_node.unsqueeze(-1).expand(-1, -1, 2)).squeeze()

        # Add the length
        length = (td['current_coord'] - current_coord).norm(p=2, dim=-1)
        length = td['length'] + length.unsqueeze(-1)

        # Update demand
        batch_size = current_node.size(0)
        demand = td['demands_with_depot']
        available_capacity = td['capacity'] - td['used_capacity']
        current_demand = torch.min(torch.gather(demand, 1, current_node), available_capacity.squeeze(-1))
        used_capacity = (td['used_capacity'] + current_demand) * (current_node != 0).float()
        demand.scatter_(1, current_node, -current_demand, reduce='add')
        
        # Update action mask
        unvisited_node = demand > 0
        unvisited_node[..., 0] = True
        action_mask = td['action_mask']
        action_mask = torch.logical_and(unvisited_node, available_capacity >= demand)
        
        # Check finish
        done = torch.logical_and(unvisited_node.sum(-1) == 0, current_node.squeeze() == 0)

        # If all nodes are visited, then set the depot be always available
        unvisited_node[..., 0] = torch.logical_or(unvisited_node[..., 0], done)
        action_mask[..., 0] = unvisited_node[..., 0]
        # action_mask[..., 0] = True

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        return TensorDict(
            {
                "next": {
                    "observation": td['observation'],
                    "depot": td['depot'],
                    "demand": td['demand'], 
                    "demands_with_depot": demand,
                    "ids": td['ids'],
                    "current_node": current_node,
                    "current_coord": current_coord,
                    "capacity": td['capacity'],
                    "used_capacity": used_capacity,
                    "length": length,
                    "unvisited_node": unvisited_node,
                    "action_mask": action_mask,
                    "i": td['i'] + 1,
                    "done": done,
                    "reward": reward,
                }
            },
            td.shape,
        )
        
    def _get_mask(self, mask, values, check_unset=True):
        assert mask.size()[:-1] == values.size()
        rng = torch.arange(mask.size(-1), out=mask.new())
        values_ = values[..., None]  # Need to broadcast up do mask dim
        # This indicates in which value of the mask a bit should be set
        where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
        # Optional: check that bit is not already set
        assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
        # Set bit by shifting a 1 to the correct position
        # (% not strictly necessary as bitshift is cyclic)
        # since where is 0 if no value needs to be set, the bitshift has no effect
        return mask | (where.long() << (values_ % 64))

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size=None
    ) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td['observation'].shape[:-2]
        device = td['observation'].device if td is not None else self.device
        self.device = device

        if td is None:
            td = self.generate_data(batch_size=batch_size).to(device)

        observation = td['observation']
        depot = td['depot']
        demand = td['demand']

        num_loc = observation.size(-2)
        return TensorDict(
            {
                "observation": torch.cat((depot[..., None, :], observation), -2),
                "depot": depot,
                "demand": demand, 
                "demands_with_depot": torch.cat((torch.zeros_like(demand[..., :1]), demand), -1),
                "ids": demand, 
                "current_node": torch.zeros((*batch_size, 1), dtype=torch.long, device=device),
                "current_coord": depot,
                "capacity": torch.full((*batch_size, 1), self.capacity, dtype=torch.float32, device=device),
                "used_capacity": demand.new_zeros((*batch_size, 1)),
                "length": demand.new_zeros((*batch_size, 1)),
                "unvisited_node": torch.ones((*batch_size, num_loc + 1), dtype=torch.bool, device=device),
                "action_mask": torch.ones((*batch_size, num_loc + 1), dtype=torch.bool, device=device),
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict = None):
        """Make the specifications of the environment (observation, action, reward, done)"""
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }
        self.observation_spec = CompositeSpec(
            observation=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            depot=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(2),
                dtype=torch.float32,
            ),
            demand=BoundedTensorSpec(
                minimum=1/CAPACITIES[self.num_loc],
                maximum=10/CAPACITIES[self.num_loc],
                shape=(self.num_loc),
                dtype=torch.float32,
            ),
            ids=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_coord=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(2),
                dtype=torch.float32,
            ),
            used_capacity=BoundedTensorSpec(
                minimum=0.,
                maximum=self.capacity,
                shape=(1),
                dtype=torch.float32,
            ),
            length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            unvisited_node=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc, 1),
                dtype=torch.bool,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc, 1),
                dtype=torch.bool,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
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
        batch_size, num_loc, _ = td['observation'].size()

        # Check if the tour is valid
        sorted_actions = actions.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        # assert (
        #     torch.arange(1, num_loc + 1, out=actions.new()).view(1, -1).expand(batch_size, num_loc) ==
        #     sorted_actions[:, -num_loc:]
        # ).all() and (sorted_actions[:, :-num_loc] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(td['demand'][:, :1], -self.capacity),
                td['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, actions)
        used_cap = torch.zeros_like(td['demand'][:, 0])
        for i in range(actions.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited

            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= self.capacity + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((td['depot'][:, None, :], td['observation']), 1)
        d = loc_with_depot.gather(1, actions[..., None].expand(*actions.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - td['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - td['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        )
    
    def generate_data(self, batch_size):
        """Dataset generation or loading"""
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }
        observation = torch.FloatTensor(*batch_size, self.num_loc, 2).uniform_(0, 1)
        demand = (torch.FloatTensor(*batch_size, self.num_loc).uniform_(0, 9).int() + 1).float() / CAPACITIES[self.num_loc]
        depot = torch.FloatTensor(*batch_size, 2).uniform_(0, 1)
        return TensorDict(
            {
                "observation": observation.to(self.device),
                "demand": demand.to(self.device),
                "depot": depot.to(self.device),
            }, 
            batch_size=batch_size
        )

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")
