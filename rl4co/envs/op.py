from typing import Optional, Union

import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# From Kool et al. 2019
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}


class OPEnv(RL4COEnvBase):
    """Orienteering Problem (OP) environment.
    At each step, the agent chooses a location to visit in order to maximize the collected prize.
    The total length of the path must not exceed a given threshold.

    Args:
        num_loc: number of locations (cities) in the OP
        min_loc: minimum value of the locations
        max_loc: maximum value of the locations
        max_length: maximum length of the path
        prize_type: type of prize to collect. Can be:
            - "dist": the prize is the distance from the previous location
            - "unif": the prize is a uniform random variable
            - "const": the prize is a constant
        td_params: parameters of the environment
    """

    name = "op"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        max_length: Union[float, torch.Tensor] = None,
        prize_type: str = "dist",
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.max_length = (
            MAX_LENGTHS.get(num_loc, None) if max_length is None else max_length
        )
        if self.max_length is None:
            raise ValueError(
                f"`max_length` must be specified for num_loc={num_loc}. Please specify it manually."
            )
        self.prize_type = prize_type
        assert self.prize_type in [
            "dist",
            "unif",
            "const",
        ], f"Invalid prize_type: {self.prize_type}"
        self._make_spec(td_params)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]

        # Update tour length
        current_loc = gather_by_index(td["locs"], current_node)
        tour_length = td["tour_length"] + (current_loc - td["current_loc"]).norm(
            p=2, dim=-1
        )

        # Update prize with collected prize
        current_total_prize = td["current_total_prize"] + gather_by_index(
            td["prize"], current_node, dim=-1
        )

        # Set current node as visited
        visited = td["visited"].scatter(-1, current_node, 1)

        # Done if went back to depot (except if it's the first step, since we start at the depot)
        done = (current_node.squeeze(-1) == 0) & (td["i"] > 0)

        # The reward is calculated outside via get_reward for efficiency, so we set it to -inf here
        reward = torch.ones_like(done) * float("-inf")

        td_step = TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "prize": td["prize"],
                    "tour_length": tour_length,
                    "current_loc": current_loc,
                    "max_length": td["max_length"],
                    "current_node": current_node,
                    "visited": visited,
                    "current_total_prize": current_total_prize,
                    "i": td["i"] + 1,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )
        td_step["next"].set("action_mask", self.get_action_mask(td_step["next"]))
        return td_step

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        # Initialize params
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        self.device = td.device

        # Add depot to locs
        locs_with_depot = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": locs_with_depot,
                "prize": F.pad(
                    td["prize"], (1, 0), mode="constant", value=0
                ),  # add 0 for depot
                "tour_length": torch.zeros(*batch_size, device=self.device),
                "current_loc": td["depot"],
                # max_length is max length allowed when arriving at node, so subtract distance to return to depot
                # Additionally, substract epsilon margin for numeric stability
                "max_length": td["max_length"][..., None]
                - (td["depot"][..., None, :] - locs_with_depot).norm(p=2, dim=-1)
                - 1e-6,
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, locs_with_depot.shape[-2]),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "current_total_prize": torch.zeros(
                    *batch_size, 1, dtype=torch.float, device=self.device
                ),
                "i": torch.zeros(
                    (*batch_size,), dtype=torch.int64, device=self.device
                ),  # counter
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Get action mask with 1 = feasible action, 0 = infeasible action.
        Cannot visit if already visited, if depot has been visited, or if the length exceeds the maximum length.
        """
        exceeds_length = (
            td["tour_length"][..., None]
            + (td["locs"] - td["current_loc"][..., None, :]).norm(p=2, dim=-1)
            > td["max_length"]
        )
        mask = td["visited"] | td["visited"][..., 0:1] | exceeds_length

        action_mask = ~mask  # 1 = feasible action, 0 = infeasible action

        # Depot can always be visited: we do not hardcode knowledge that this is strictly suboptimal if other options are available
        action_mask[..., 0] = 1
        return action_mask

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Reward is the sum of the prizes of visited nodes"""

        # In case all tours directly return to depot, prevent further problems
        if actions.size(-1) == 1:
            assert (actions == 0).all(), "If all length 1 tours, they should be zero"
            return torch.zeros(actions.size(0), dtype=torch.float, device=actions.device)

        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Prize is the sum of the prizes of the visited nodes. Note that prize is padded with 0 for depot at index 0
        collected_prize = td["prize"].gather(1, actions)
        return collected_prize.sum(-1)

    @staticmethod
    def check_solution_validity(
        td: TensorDict, actions: torch.Tensor, add_distance_to_depot: bool = True
    ):
        """Check that solution is valid: nodes are not visited twice except depot and capacity is not exceeded.
        If `add_distance_to_depot` if True, then the distance to the depot is added to max length since by default, the max length is
        modified in the reset function to account for the distance to the depot.
        """

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_actions = actions.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_actions[:, 1:] == 0)
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"

        # Gather locations in order of tour and get the length of tours
        locs_ordered = gather_by_index(td["locs"], actions)
        length = get_tour_length(locs_ordered)

        max_length = td["max_length"]
        if add_distance_to_depot:
            max_length = (
                max_length
                + (td["locs"][..., 0:1, :] - td["locs"]).norm(p=2, dim=-1)
                + 1e-6
            )

        assert (
            length[..., None] <= max_length + 1e-5
        ).all(), "Max length exceeded by {}".format(
            (length[..., None] - max_length).max()
        )

    def generate_data(self, batch_size, prize_type=None) -> TensorDict:
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        prize_type = self.prize_type if prize_type is None else prize_type

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Methods taken from Fischetti et al. (1998) and Kool et al. (2019)
        if prize_type == "const":
            prize = torch.ones(*batch_size, self.num_loc, device=self.device)
        elif prize_type == "unif":
            prize = (
                1
                + torch.randint(
                    0, 100, (*batch_size, self.num_loc), device=self.device
                ).float()
            ) / 100
        elif prize_type == "dist":  # based on the distance to the depot
            prize = (locs_with_depot[..., 0:1, :] - locs_with_depot[..., 1:, :]).norm(
                p=2, dim=-1
            )
            prize = (
                1 + (prize / prize.max(dim=-1, keepdim=True)[0] * 99).int()
            ).float() / 100
        else:
            raise ValueError(f"Invalid prize_type: {self.prize_type}")

        # Support for heterogeneous max length if provided
        if not isinstance(self.max_length, torch.Tensor):
            max_length = torch.full((*batch_size,), self.max_length, device=self.device)
        else:
            max_length = self.max_length

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "prize": prize,
                "max_length": max_length,
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
            prize=UnboundedContinuousTensorSpec(
                shape=(self.num_loc,),
                dtype=torch.float32,
            ),
            tour_length=UnboundedContinuousTensorSpec(
                shape=(self.num_loc,),
                dtype=torch.float32,
            ),
            visited=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc + 1,),
                dtype=torch.bool,
            ),
            current_loc=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(2,),
                dtype=torch.float32,
            ),
            max_length=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc + 1, 1),
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
    def render(td: TensorDict, actions=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a plot of the nodes
        if ax is None:
            _, ax = plt.subplots()

        td = td.detach().cpu()

        # Actions
        if actions is None:
            actions = td.get("action", None)
        actions = actions.detach().cpu() if actions is not None else None

        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0] if actions is not None else None

        # Variables
        depot = td["locs"][0, :]
        cities = td["locs"][1:, :]
        prizes = td["prize"][1:]
        normalized_prizes = (
            200 * (prizes - torch.min(prizes)) / (torch.max(prizes) - torch.min(prizes))
            + 10
        )

        # Plot depot and cities with prize
        ax.scatter(
            depot[0],
            depot[1],
            marker="s",
            c="tab:green",
            edgecolors="black",
            zorder=5,
            s=100,
        )  # Plot depot as square
        ax.scatter(
            cities[:, 0],
            cities[:, 1],
            s=normalized_prizes,
            c=normalized_prizes,
            cmap="autumn_r",
            alpha=0.6,
            edgecolors="black",
        )  # Plot all cities with size and color indicating the prize

        # Gather locs in order of action if available
        if actions is None:
            log.warning("No action in TensorDict, rendering unsorted locs")
        else:
            # Reorder the cities and their corresponding prizes based on actions
            tour = cities[actions - 1]  # subtract 1 to match Python's 0-indexing

            # Append the depot at the beginning and the end of the tour
            tour = np.vstack((depot, tour, depot))

            # Use quiver to plot the tour
            dx, dy = np.diff(tour[:, 0]), np.diff(tour[:, 1])
            ax.quiver(
                tour[:-1, 0],
                tour[:-1, 1],
                dx,
                dy,
                scale_units="xy",
                angles="xy",
                scale=1,
                zorder=2,
                color="black",
                width=0.0035,
            )

        # Setup limits and show
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.show()
