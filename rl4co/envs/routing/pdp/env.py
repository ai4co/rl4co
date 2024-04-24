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
from rl4co.utils.ops import gather_by_index, get_tour_length

from .generator import PDPGenerator
from .render import render


class PDPEnv(RL4COEnvBase):
    """Pickup and Delivery Problem (PDP) environment.
    The goal is to visit all the pickup and delivery locations in the shortest path possible starting from the depot
    The conditions is that the agent must visit a pickup location before visiting its corresponding delivery location

    Observations:
        - locations of the depot, pickup, and delivery locations
        - current location of the vehicle
        - the remaining locations to deliver
        - the visited locations
        - the current step

    Constraints:
        - the tour starts and ends at the depot
        - each pickup location must be visited before its corresponding delivery location
        - the vehicle cannot visit the same location twice

    Finish Condition:
        - the vehicle has visited all locations

    Reward:
        - (minus) the negative length of the path

    Args:
        generator: PDPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "pdp"

    def __init__(
        self,
        generator: PDPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = PDPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"].unsqueeze(-1)

        num_loc = td["locs"].shape[-2] - 1  # except depot

        # Pickup and delivery node pair of selected node
        new_to_deliver = (current_node + num_loc // 2) % (num_loc + 1)

        # Set visited to 0 (i.e., we visited the node)
        visited = td["visited"].scatter(
            -1, current_node.expand_as(td["action_mask"]), 1
        )

        to_deliver = td["to_deliver"].scatter(
            -1, new_to_deliver.expand_as(td["to_deliver"]), 1
        )

        # Action is feasible if the node is not visited and is to deliver
        # action_mask = torch.logical_and(visited, to_deliver)
        action_mask = ~visited & to_deliver

        # We are done there are no unvisited locations
        done = visited.sum(-1, keepdim=True) == visited.size(-1)
        reward = torch.zeros_like(done)

        # Update step
        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "to_deliver": to_deliver,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None) -> TensorDict:
        device = td.device
        locs = td["locs"]
        num_loc = locs.size(-2)

        # Mark pickup with 1, deliver with 0, e.g. [1, 1, ..., 1, 0, ..., 0]
        to_deliver = torch.cat(
            [
                torch.ones((*batch_size,num_loc // 2 + 1), dtype=torch.bool, device=device),
                torch.zeros((*batch_size, num_loc // 2), dtype=torch.bool, device=device),
            ], dim=-1
        )

        # Depot and all delivery nodes are not available at the beginning
        action_mask = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        action_mask[:, 0] = False 
        action_mask[:, num_loc // 2 + 1:] = False
        
        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        visited = torch.zeros((*batch_size, num_loc), dtype=torch.bool, device=device)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        done = torch.zeros((*batch_size, 1), dtype=torch.bool, device=device)

        # Depot is always visited
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "to_deliver": to_deliver,
                "visited": visited,
                "i": i,
                "action_mask": action_mask,
                "done": done,
            },
            batch_size=batch_size,
        )
        return td_reset

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> torch.Tensor:
        # Gather locations in the order of actions and get reward = -(total distance)
        locs_ordered = gather_by_index(td["locs"], actions)  # [batch, graph_size+1, 2]
        return -get_tour_length(locs_ordered)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # Append depot to the beginning of the tour
        actions = torch.cat((torch.zeros_like(actions[:, :1]), actions), dim=-1)
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Not visiting all nodes"

        visited_time = torch.argsort(
            actions, 1
        )  # index of pickup less than index of delivery
        assert (
            visited_time[:, 1 : actions.size(1) // 2 + 1]
            < visited_time[:, actions.size(1) // 2 + 1 :]
        ).all(), "Deliverying without pick-up"

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor=None, ax = None):
        return render(td, actions, ax)

    def _make_spec(self, generator: PDPGenerator):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            to_deliver=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
