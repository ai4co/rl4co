from typing import Optional

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index, get_distance, get_tour_length


class MTSPEnv(RL4COEnvBase):
    """Multiple Traveling Salesman Problem environment
    At each step, an agent chooses to visit a city. A maximum of `num_agents` agents can be employed to visit the cities.
    The cost can be defined in two ways:
        - `minmax`: (default) the reward is the maximum of the path lengths of all the agents
        - `sum`: the cost is the sum of the path lengths of all the agents
    Reward is - cost, so the goal is to maximize the reward (minimize the cost).

    Args:
        num_loc: number of locations (cities) to visit
        min_loc: minimum value of the locations
        max_loc: maximum value of the locations
        min_num_agents: minimum number of agents
        max_num_agents: maximum number of agents
        cost_type: type of cost to use, either `minmax` or `sum`
        td_params: parameters for the TensorDict specs
    """

    name = "mtsp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_num_agents: int = 5,
        max_num_agents: int = 5,
        cost_type: str = "minmax",
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents
        self.cost_type = cost_type
        self._make_spec(td_params)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        # Initial variables
        is_first_action = batch_to_scalar(td["i"]) == 0
        current_node = td["action"]
        first_node = current_node if is_first_action else td["first_node"]

        # Get the locations of the current node and the previous node and the depot
        cur_loc = gather_by_index(td["locs"], current_node)
        prev_loc = gather_by_index(
            td["locs"], td["current_node"]
        )  # current_node is the previous node
        depot_loc = td["locs"][..., 0, :]

        # If current_node is the depot, then increment agent_idx
        cur_agent_idx = td["agent_idx"] + (current_node == 0).long()

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node[..., None].expand_as(td["action_mask"]), 0
        )
        # Available[..., 0] is the depot, which is always available unless:
        # - current_node is the depot
        # - agent_idx greater than num_agents -1
        available[..., 0] = torch.logical_and(
            current_node != 0, td["agent_idx"] < td["num_agents"] - 1
        )

        # We are done there are no unvisited locations except the depot
        done = torch.count_nonzero(available[..., 1:], dim=-1) == 0

        # If done is True, then we make the depot available again, so that it will be selected as the next node with prob 1
        available[..., 0] = torch.logical_or(done, available[..., 0])

        # If current agent is different from previous agent, then we have a new subtour and reset the length, otherwise we add the new distance
        current_length = torch.where(
            cur_agent_idx != td["agent_idx"],
            torch.zeros_like(td["current_length"]),
            td["current_length"] + get_distance(cur_loc, prev_loc),
        )

        # If done, we add the distance from the current_node to the depot as well
        current_length = torch.where(
            done, current_length + get_distance(cur_loc, depot_loc), current_length
        )

        # We update the max_subtour_length and reset the current_length
        max_subtour_length = torch.where(
            current_length > td["max_subtour_length"],
            current_length,
            td["max_subtour_length"],
        )

        # The reward is the negative of the max_subtour_length (minmax objective)
        reward = -max_subtour_length

        # The output must be written in a ``"next"`` entry
        return TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "num_agents": td["num_agents"],
                    "max_subtour_length": max_subtour_length,
                    "current_length": current_length,
                    "agent_idx": cur_agent_idx,
                    "first_node": first_node,
                    "current_node": current_node,
                    "i": td["i"] + 1,
                    "action_mask": available,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize data
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]

        device = td.device if td is not None else self.device
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        # Keep track of the agent number to know when to stop
        agent_idx = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        # Make variable for max_subtour_length between subtours
        max_subtour_length = torch.zeros(batch_size, dtype=torch.float32, device=device)
        current_length = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # Other variables
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        available[..., 0] = 0  # Depot is not available as first node
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": td["locs"],  # depot is first node
                "num_agents": td["num_agents"],
                "max_subtour_length": max_subtour_length,
                "current_length": current_length,
                "agent_idx": agent_idx,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc, 2),
                dtype=torch.float32,
            ),
            num_agents=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            agent_idx=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            max_subtour_length=UnboundedContinuousTensorSpec(
                shape=(1),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
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
        self.reward_spec = UnboundedContinuousTensorSpec()
        self.done_spec = UnboundedDiscreteTensorSpec(dtype=torch.bool)

    def get_reward(self, td, actions=None) -> TensorDict:
        # With minmax, get the maximum distance among subtours, calculated in the model
        if self.cost_type == "minmax":
            return td["reward"].squeeze(-1)

        # With distance, same as TSP
        elif self.cost_type == "distance":
            locs = td["locs"]
            locs_ordered = locs.gather(1, actions.unsqueeze(-1).expand_as(locs))
            return -get_tour_length(locs_ordered)

        else:
            raise ValueError(f"Cost type {self.cost_type} not supported")

    def generate_data(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs = (
            torch.FloatTensor(*batch_size, self.num_loc, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the num_agents: either fixed or random integer between min and max
        if self.min_num_agents == self.max_num_agents:
            num_agents = (
                torch.ones(batch_size, dtype=torch.int64, device=self.device)
                * self.min_num_agents
            )
        else:
            num_agents = torch.randint(
                self.min_num_agents,
                self.max_num_agents,
                size=batch_size,
                device=self.device,
            )

        return TensorDict(
            {
                "locs": locs,
                "num_agents": num_agents,
            },
            batch_size=batch_size,
        )

    @staticmethod
    def render(td, actions=None, ax=None):
        import matplotlib.pyplot as plt

        from matplotlib import colormaps

        def discrete_cmap(num, base_cmap="nipy_spectral"):
            """Create an N-bin discrete colormap from the specified input map"""
            base = colormaps[base_cmap]
            color_list = base(np.linspace(0, 1, num))
            cmap_name = base.name + str(num)
            return base.from_list(cmap_name, color_list, num)

        if actions is None:
            actions = td.get("action", None)
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]

        num_agents = td["num_agents"]
        locs = td["locs"]
        cmap = discrete_cmap(num_agents, "rainbow")

        fig, ax = plt.subplots()

        # Add depot action = 0 to before first action and after last action
        actions = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64),
                actions,
                torch.zeros(1, dtype=torch.int64),
            ]
        )

        # Make list of colors from matplotlib
        for i, loc in enumerate(locs):
            if i == 0:
                # depot
                marker = "s"
                color = "g"
                label = "Depot"
                markersize = 10
            else:
                # normal location
                marker = "o"
                color = "tab:blue"
                label = "Cities"
                markersize = 8
            if i > 1:
                label = ""

            ax.plot(
                loc[0],
                loc[1],
                color=color,
                marker=marker,
                markersize=markersize,
                label=label,
            )

        # Plot the actions in order
        agent_idx = 0
        for i in range(len(actions)):
            if actions[i] == 0:
                agent_idx += 1
            color = cmap(num_agents - agent_idx)

            from_node = actions[i]
            to_node = (
                actions[i + 1] if i < len(actions) - 1 else actions[0]
            )  # last goes back to depot
            from_loc = td["locs"][from_node]
            to_loc = td["locs"][to_node]
            ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color)
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="->", color=color),
                annotation_clip=False,
            )

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        ax.set_title("mTSP")
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        plt.show()
