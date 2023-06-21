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

from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import gather_by_index

# For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
# so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
# of the nodes by half of the tour length (which is very rough but similar to op)
# This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
# The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
# is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
# However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}


class PCTSPEnv(RL4COEnvBase):
    """Prize-collecting TSP environment
    The goal is to collect as much prize as possible while minimizing the total travel cost
    The environment is stochastic, the prize is only revealed when the node is visited

    Args:
        num_loc (int): Number of locations
        min_loc (float): Minimum location value
        max_loc (float): Maximum location value
        penalty_factor (float): Penalty factor
        prize_required (float): Minimum prize required to visit a node
        stochastic (bool): Whether the environment is stochastic
        td_params (TensorDict): Parameters of the environment
    """

    name = "pctsp"

    def __init__(
        self,
        num_loc: int = 10,
        min_loc: float = 0,
        max_loc: float = 1,
        penalty_factor: float = 3,
        prize_required: float = 1,
        stochastic: bool = False,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.penalty_factor = penalty_factor
        self.prize_required = prize_required
        self.stochastic = stochastic

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]

        # Get current coordinates, prize, and penalty
        cur_total_prize = td["cur_total_prize"] + gather_by_index(
            td["real_prize"], current_node
        )
        cur_total_penalty = td["cur_total_penalty"] + gather_by_index(
            td["penalty"], current_node
        )

        # Update masks
        visited = td["visited"].scatter(-1, current_node[..., None], 1)
        mask = visited | visited[..., 0:1]

        # Cannot visit depot if not yet collected 1 total prize and there are unvisited nodes
        mask[..., 0] = (cur_total_prize < 1.0) & (
            visited[..., 1:].int().sum(-1) < visited[..., 1:].size(-1)
        )
        action_mask = ~(mask > 0)  # Invert mask

        # Done and reward. Calculation is done outside hence set -inf
        done = (td["i"] > 0) & (current_node == 0)
        reward = torch.ones_like(cur_total_prize) * float("-inf")

        return TensorDict(
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
                    "action_mask": action_mask,
                    "reward": reward,
                    "done": done,
                },
            },
            batch_size=td.batch_size,
        )

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
        real_prize = td["stochastic_prize"] if self.stochastic else expected_prize
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

        # Cannot visit depot if not yet collected 1 total prize and there are unvisited nodes
        mask = visited | visited[..., 0:1]
        mask[..., 0] = (cur_total_prize < 1.0) & (
            visited[..., 1:].int().sum(-1) < visited[..., 1:].size(-1)
        )
        action_mask = ~(mask > 0)  # Invert mask

        return TensorDict(
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
                "action_mask": action_mask,
            },
            batch_size=batch_size,
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

    def get_reward(self, td, actions):
        # if actions.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
        #     assert (actions == 0).all(), "If all length 1 tours, they should be zero"
        #     return torch.zeros(actions.size(0), dtype=torch.float, device=actions.device)

        # Check that tours are valid, i.e. contain 0 to n -1
        # sorted_actions = actions.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        # assert ((sorted_actions[..., 1:] == 0) | (sorted_actions[..., 1:] > sorted_actions[..., :-1])).all(), "Duplicates"

        prize = td["real_prize"][..., 1:]  # Remove depot
        prize_with_depot = torch.cat((torch.zeros_like(prize[:, :1]), prize), 1)
        prize_with_depot.gather(1, actions)

        locs_with_depot = td["locs"]
        depot = locs_with_depot[..., 0, :]
        td["locs"][..., 1:]  # Remove depot

        # Either prize constraint should be satisfied or all prizes should be visited
        # assert (
        #     (p.sum(-1) >= 1 - 1e-5) |
        #     (sorted_actions.size(-1) - (sorted_actions == 0).int().sum(-1) == locs.size(-2)) # no depot
        # ).all(), "Total prize does not satisfy min total prize"

        pen = td["penalty"].gather(1, actions)

        # Gather td in order of tour. We consider locs already have depot concatenated
        d = gather_by_index(locs_with_depot, actions)

        length = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
            + (d[:, 0] - depot).norm(p=2, dim=-1)  # Depot to first
            + (d[:, -1] - depot).norm(
                p=2, dim=-1
            )  # Last to depot, will be 0 if depot is last
        )
        # We want to maximize total prize
        # Incurred penalty cost is total penalty cost - saved penalty costs of nodes visited
        return -(length + td["penalty"][..., 1:].sum(-1) - pen.sum(-1))

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

    def render(self, td: TensorDict):
        raise NotImplementedError("TODO: render is not implemented yet")
