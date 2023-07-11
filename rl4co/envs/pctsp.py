from typing import Optional

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

# For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
# so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
# of the nodes by half of the tour length (which is very rough but similar to op)
# This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
# The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
# is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
# However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}


class PCTSPEnv(RL4COEnvBase):
    """Prize-collecting TSP (PCTSP) environment.
    The goal is to collect as much prize as possible while minimizing the total travel cost.
    The environment is stochastic, the prize is only revealed when the node is visited.

    Args:
        num_loc: Number of locations
        min_loc: Minimum location value
        max_loc: Maximum location value
        penalty_factor: Penalty factor
        prize_required: Minimum prize required to visit a node
        check_solution: Set to False by default for small bug happening around 0.01% of the time (TODO: fix)
        td_params: Parameters of the environment
    """

    name = "pctsp"
    _stochastic = False

    def __init__(
        self,
        num_loc: int = 10,
        min_loc: float = 0,
        max_loc: float = 1,
        penalty_factor: float = 3,
        prize_required: float = 1,
        check_solution: bool = False,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.penalty_factor = penalty_factor
        self.prize_required = prize_required
        self.check_solution = check_solution

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]

        # Get current coordinates, prize, and penalty
        cur_total_prize = td["cur_total_prize"] + gather_by_index(
            td["real_prize"], current_node
        )
        cur_total_penalty = td["cur_total_penalty"] + gather_by_index(
            td["penalty"], current_node
        )

        # Update visited
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # Done and reward. Calculation is done outside hence set -inf
        done = (td["i"] > 0) & (current_node == 0)
        reward = torch.ones_like(cur_total_prize) * float("-inf")

        td_step = TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "current_node": current_node,
                    "expected_prize": td["expected_prize"],
                    "real_prize": td["real_prize"],
                    "penalty": td["penalty"],
                    "cur_total_prize": cur_total_prize,
                    "cur_total_penalty": cur_total_penalty,
                    "visited": visited,
                    "prize_required": td["prize_required"],
                    "i": td["i"] + 1,
                    "reward": reward,
                    "done": done,
                },
            },
            batch_size=td.batch_size,
        )
        td_step["next"].set("action_mask", self.get_action_mask(td_step["next"]))
        return td_step

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        self.device = td.device

        locs = torch.cat([td["depot"][..., None, :], td["locs"]], dim=-2)
        expected_prize = td["deterministic_prize"]
        real_prize = (
            td["stochastic_prize"] if self.stochastic else td["deterministic_prize"]
        )
        penalty = td["penalty"]

        # Concatenate depots
        real_prize_with_depot = torch.cat(
            [torch.zeros_like(real_prize[..., :1]), real_prize], dim=-1
        )
        penalty_with_depot = F.pad(penalty, (1, 0), mode="constant", value=0)

        # Initialize the current node and  prize / penalty
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=self.device)
        cur_total_prize = torch.zeros(*batch_size, device=self.device)
        cur_total_penalty = penalty.sum(-1)[
            :, None
        ]  # Sum penalties (all when nothing is visited), add step dim

        # Init the action mask (all nodes are available)
        visited = torch.zeros(
            (*batch_size, self.num_loc + 1), dtype=torch.bool, device=self.device
        )
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=self.device)
        prize_required = torch.full(
            (*batch_size,), self.prize_required, device=self.device
        )

        td_reset = TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "expected_prize": expected_prize,
                "real_prize": real_prize_with_depot,
                "penalty": penalty_with_depot,
                "cur_total_prize": cur_total_prize,
                "cur_total_penalty": cur_total_penalty,
                "visited": visited,
                "prize_required": prize_required,
                "i": i,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Cannot visit depot if not yet collected 1 total prize and there are unvisited nodes"""
        mask = td["visited"] | td["visited"][..., 0:1]
        mask[..., 0] = (td["cur_total_prize"] < 1.0) & (
            td["visited"][..., 1:].int().sum(-1) < td["visited"][..., 1:].size(-1)
        )
        return ~(mask > 0)  # Invert mask, since 1 means feasible action

    def get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Reward is `saved penalties - (total length + penalty)`"""

        # In case all tours directly return to depot, prevent further problems
        if actions.size(-1) == 1:
            assert (actions == 0).all(), "If all length 1 tours, they should be zero"
            return torch.zeros(actions.size(0), dtype=torch.float, device=actions.device)

        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather locations in order of tour and get the length of tours
        locs_ordered = gather_by_index(td["locs"], actions)
        length = get_tour_length(locs_ordered)

        # Reward is saved penalties - (total length + penalty)
        saved_penalty = td["penalty"].gather(1, actions)
        return saved_penalty.sum(-1) - (length + td["penalty"][..., 1:].sum(-1))

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that the solution is valid, i.e. contains all nodes once at most, and either prize constraint is met or all nodes are visited"""

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_actions = actions.data.sort(1)[0]

        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_actions[..., 1:] == 0)
            | (sorted_actions[..., 1:] > sorted_actions[..., :-1])
        ).all(), "Duplicates"

        prize = td["real_prize"][..., 1:]  # Remove depot
        prize_with_depot = torch.cat((torch.zeros_like(prize[:, :1]), prize), 1)
        p = prize_with_depot.gather(1, actions)

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
            (p.sum(-1) >= 1 - 1e-5)
            | (
                sorted_actions.size(-1) - (sorted_actions == 0).int().sum(-1)
                == (td["locs"].size(-2) - 1)
            )  # no depot
        ).all(), "Total prize does not satisfy min total prize"

    def generate_data(self, batch_size) -> TensorDict:
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        depot = torch.rand((*batch_size, 2))
        locs = torch.rand((*batch_size, self.num_loc, 2))

        penalty_max = (
            MAX_LENGTHS[self.num_loc] * (self.penalty_factor) / float(self.num_loc)
        )
        penalty = torch.rand((*batch_size, self.num_loc)) * penalty_max

        # Take uniform prizes
        # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
        # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
        # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
        deterministic_prize = (
            torch.rand((*batch_size, self.num_loc)) * 4 / float(self.num_loc)
        )

        # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
        # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
        # stochastic prize is only revealed once the node is visited
        # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
        stochastic_prize = (
            torch.rand((*batch_size, self.num_loc)) * deterministic_prize * 2
        )
        # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
        # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
        # stochastic prize is only revealed once the node is visited
        # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
        stochastic_prize = (
            torch.rand((*batch_size, self.num_loc)) * deterministic_prize * 2
        )

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "penalty": penalty,
                "deterministic_prize": deterministic_prize,
                "stochastic_prize": stochastic_prize,
            },
            batch_size=batch_size,
        )

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is True:
            log.warning(
                "Stochastic mode should not be used for PCTSP. Use SPCTSP instead."
            )

    def _make_spec(self, td_params: TensorDict):
        """Make the locs and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            expected_prize=UnboundedContinuousTensorSpec(
                shape=(self.num_loc),
                dtype=torch.float32,
            ),
            real_prize=UnboundedContinuousTensorSpec(
                shape=(self.num_loc + 1),
                dtype=torch.float32,
            ),
            penalty=UnboundedContinuousTensorSpec(
                shape=(self.num_loc + 1),
                dtype=torch.float32,
            ),
            cur_total_prize=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            cur_total_penalty=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            visited=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc + 1),
                dtype=torch.bool,
            ),
            prize_required=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
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
    def render(td, actions=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import colormaps

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
        prizes = td["real_prize"][1:]
        penalties = td["penalty"][1:]
        normalized_prizes = (
            200 * (prizes - torch.min(prizes)) / (torch.max(prizes) - torch.min(prizes))
            + 10
        )
        normalized_penalties = (
            3
            * (penalties - torch.min(penalties))
            / (torch.max(penalties) - torch.min(penalties))
        )

        # Represent penalty with colormap and size of edges
        penalty_cmap = colormaps.get_cmap("BuPu")
        penalty_colors = penalty_cmap(normalized_penalties)

        # Plot depot and cities with prize (size of nodes) and penalties (size of borders)
        ax.scatter(
            depot[0],
            depot[1],
            marker="s",
            c="tab:green",
            edgecolors="black",
            zorder=1,
            s=100,
        )  # Plot depot as square
        ax.scatter(
            cities[:, 0],
            cities[:, 1],
            s=normalized_prizes,
            c=normalized_prizes,
            cmap="autumn_r",
            alpha=1,
            edgecolors=penalty_colors,
            linewidths=normalized_penalties,
        )  # Plot all cities with size and color indicating the prize

        # Gather locs in order of action if available
        if actions is None:
            print("No action in TensorDict, rendering unsorted locs")
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
