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
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self._make_spec(td_params)

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

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "current_node": current_node,
                    "available": available,
                    "to_deliver": to_deliver,
                    "i": td["i"] + 1,
                    "action_mask": action_mask,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.device = td.device

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Pick is 1, deliver is 0 [batch_size, graph_size+1], [1,1...1, 0...0]
        to_deliver = torch.cat(
            [
                torch.ones(
                    *batch_size,
                    self.num_loc // 2 + 1,
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.zeros(
                    *batch_size, self.num_loc // 2, dtype=torch.bool, device=self.device
                ),
            ],
            dim=-1,
        )

        # Cannot visit depot at first step # [0,1...1] so set not available
        available = torch.ones(
            (*batch_size, self.num_loc + 1), dtype=torch.bool, device=self.device
        )
        action_mask = ~available.contiguous()  # [batch_size, graph_size+1]
        action_mask[..., 0] = 1  # First step is always the depot

        # Other variables
        current_node = torch.zeros(
            (*batch_size, 1), dtype=torch.int64, device=self.device
        )
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=self.device)

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

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc + 1, 2),
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
                shape=(self.num_loc + 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
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

        # Gather locations in the order of actions and get reward = -(total distance)
        locs_ordered = gather_by_index(td["locs"], actions)  # [batch, graph_size+1, 2]
        return -get_tour_length(locs_ordered)

    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
            },
            batch_size=batch_size,
        )

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        import matplotlib.pyplot as plt

        markersize = 8

        td = td.detach().cpu()
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            if actions is not None:
                actions = actions[0]

        # Variables
        init_deliveries = td["to_deliver"][1:]
        delivery_locs = td["locs"][1:][~init_deliveries.bool()]
        pickup_locs = td["locs"][1:][init_deliveries.bool()]
        depot_loc = td["locs"][0]
        actions = actions if actions is not None else td["action"]

        fig, ax = plt.subplots()

        # Plot the actions in order
        for i in range(len(actions)):
            from_node = actions[i]
            to_node = (
                actions[i + 1] if i < len(actions) - 1 else actions[0]
            )  # last goes back to depot
            from_loc = td["locs"][from_node]
            to_loc = td["locs"][to_node]
            ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], "k-")
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="->", color="black"),
                annotation_clip=False,
            )

        # Plot the depot location
        ax.plot(
            depot_loc[0],
            depot_loc[1],
            "g",
            marker="s",
            markersize=markersize,
            label="Depot",
        )

        # Plot the pickup locations
        for i, pickup_loc in enumerate(pickup_locs):
            ax.plot(
                pickup_loc[0],
                pickup_loc[1],
                "r",
                marker="^",
                markersize=markersize,
                label="Pickup" if i == 0 else None,
            )

        # Plot the delivery locations
        for i, delivery_loc in enumerate(delivery_locs):
            ax.plot(
                delivery_loc[0],
                delivery_loc[1],
                "b",
                marker="v",
                markersize=markersize,
                label="Delivery" if i == 0 else None,
            )

        # Setup limits and show
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.show()
