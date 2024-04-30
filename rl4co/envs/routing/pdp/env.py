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
    The environment is made of num_loc + 1 locations (cities):
        - 1 depot
        - `num_loc` / 2 pickup locations
        - `num_loc` / 2 delivery locations
    The goal is to visit all the pickup and delivery locations in the shortest path possible starting from the depot
    The conditions is that the agent must visit a pickup location before visiting its corresponding delivery location

    Args:
        num_loc: number of locations (cities) in the TSP
        td_params: parameters of the environment
        seed: seed for the environment
        device: device to use.  Generally, no need to set as tensors are updated on the fly
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

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"].unsqueeze(-1)

        num_loc = td["locs"].shape[-2] - 1  # except depot

        # Pickup and delivery node pair of selected node
        new_to_deliver = (current_node + num_loc // 2) % (num_loc + 1)

        # Set available to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node.expand_as(td["action_mask"]), 0
        )

        to_deliver = td["to_deliver"].scatter(
            -1, new_to_deliver.expand_as(td["to_deliver"]), 1
        )

        # Action is feasible if the node is not visited and is to deliver
        # action_mask = torch.logical_and(available, to_deliver)
        action_mask = available & to_deliver

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        # Update step
        td.update(
            {
                "current_node": current_node,
                "available": available,
                "to_deliver": to_deliver,
                "i": td["i"] + 1,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Pick is 1, deliver is 0 [batch_size, graph_size+1], [1,1...1, 0...0]
        to_deliver = torch.cat(
            [
                torch.ones(
                    *batch_size,
                    self.generator.num_loc // 2 + 1,
                    dtype=torch.bool,
                    device=device,
                ),
                torch.zeros(
                    *batch_size, self.generator.num_loc // 2, dtype=torch.bool, device=device
                ),
            ],
            dim=-1,
        )

        # Cannot visit depot at first step # [0,1...1] so set not available
        available = torch.ones(
            (*batch_size, self.generator.num_loc + 1), dtype=torch.bool, device=device
        )
        action_mask = ~available.contiguous()  # [batch_size, graph_size+1]
        action_mask[..., 0] = 1  # First step is always the depot

        # Other variables
        current_node = torch.zeros(
            (*batch_size, 1), dtype=torch.int64, device=device
        )
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "to_deliver": to_deliver,
                "available": available,
                "i": i,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

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

    @staticmethod
    def _get_reward(td, actions) -> TensorDict:
        # Gather locations in the order of actions and get reward = -(total distance)
        locs_ordered = gather_by_index(td["locs"], actions)  # [batch, graph_size+1, 2]
        return -get_tour_length(locs_ordered)

    def check_solution_validity(self, td, actions):
        # assert (actions[:, 0] == 0).all(), "Not starting at depot"
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
