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
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MPDPEnv(RL4COEnvBase):
    """Multi-agent Pickup and Delivery Problem environment.
    The goal is to pick up and deliver all the packages while satisfying the precedence constraints.
    When an agent goes back to the depot, a new agent is spawned. In the min-max version, the goal is to minimize the
    maximum tour length among all agents.
    The reward is the -infinite unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Args:
        num_loc: number of locations (cities) in the TSP
        min_loc: minimum location coordinate. Used for data generation
        max_loc: maximum location coordinate. Used for data generation
        min_num_agents: minimum number of agents. Used for data generation
        max_num_agents: maximum number of agents. Used for data generation
        objective: objective to optimize. Either 'minmax' or 'minsum'
        check_solution: whether to check the validity of the solution
        td_params: parameters of the environment
    """

    name = "mpdp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_num_agents: int = 2,
        max_num_agents: int = 10,
        objective: str = "minmax",
        check_solution: bool = False,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_num_agents = min_num_agents
        self.max_num_agents = max_num_agents
        self.objective = objective
        self.check_solution = check_solution
        self._make_spec(td_params)

    def _step(self, td: TensorDict) -> TensorDict:
        selected = td["action"][:, None]  # Add dimension for step

        agent_num = td["lengths"].size(1)
        n_loc = td["to_delivery"].size(-1) - agent_num - 1

        new_to_delivery = (selected + n_loc // 2) % (
            n_loc + agent_num + 1
        )  # the pair node of selected node

        is_request = (selected > agent_num) & (selected <= agent_num + n_loc // 2)
        td["left_request"][is_request] -= 1
        depot_distance = td["depot_distance"].scatter(-1, selected, 0)

        add_pd = td["add_pd_distance"][is_request.squeeze(-1), :].gather(
            -1, selected[is_request.squeeze(-1), :] - agent_num - 1
        )
        td["longest_lengths"][is_request.squeeze(-1), :].scatter_add_(
            -1, td["count_depot"][is_request.squeeze(-1), :], add_pd
        )
        td["add_pd_distance"][is_request.squeeze(-1), :].scatter_(
            -1, selected[is_request.squeeze(-1), :] - agent_num - 1, 0
        )
        remain_sum_paired_distance = td["add_pd_distance"].sum(-1, keepdim=True)
        remain_pickup_max_distance = depot_distance[:, : agent_num + 1 + n_loc // 2].max(
            dim=-1, keepdim=True
        )[0]
        remain_delivery_max_distance = depot_distance[
            :, agent_num + 1 + n_loc // 2 :
        ].max(dim=-1, keepdim=True)[0]

        # Calculate makespan
        cur_coord = gather_by_index(td["locs"], selected)
        path_lengths = (cur_coord - td["cur_coord"]).norm(p=2, dim=-1)

        td["lengths"].scatter_add_(-1, td["count_depot"], path_lengths.unsqueeze(-1))

        # If visit depot then plus one to count_depot\
        td["count_depot"][
            (selected == td["agent_idx"]) & (td["agent_idx"] < agent_num)
        ] += 1  # torch.ones(td["count_depot"][(selected == 0) & (td["agent_idx"] < agent_num)].shape, dtype=torch.int64, device=td["count_depot"].device)

        # `agent_idx` is added by 1 if the current agent comes back to depot
        agent_idx = (td["count_depot"] + 1) * torch.ones(
            selected.size(0), 1, dtype=torch.long, device=td["count_depot"].device
        )
        visited = td["visited"].scatter(-1, selected.unsqueeze(-1), 1)
        to_delivery = td["to_delivery"].scatter(-1, new_to_delivery[:, :, None], 1)

        # Get done and reward
        done = visited.all(dim=-1, keepdim=True).squeeze(-1)
        reward = torch.ones_like(done) * float(
            "-inf"
        )  # reward calculated via `get_reward` for now

        td_step = TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "visited": visited,
                    "lengths": td["lengths"],
                    "count_depot": td["count_depot"],
                    "agent_idx": agent_idx,
                    "cur_coord": cur_coord,
                    "to_delivery": to_delivery,
                    "left_request": td["left_request"],
                    "depot_distance": depot_distance,
                    "remain_sum_paired_distance": remain_sum_paired_distance,
                    "remain_pickup_max_distance": remain_pickup_max_distance,
                    "remain_delivery_max_distance": remain_delivery_max_distance,
                    "add_pd_distance": td["add_pd_distance"],
                    "longest_lengths": td["longest_lengths"],
                    "i": td["i"] + 1,
                    "done": done,
                    "reward": reward,
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
        agent_num: Optional[int] = None,  # NOTE hardcoded from ET
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.device = td.device

        # NOTE: this is a hack to get the agent_num
        # agent_num = td["agent_num"][0].item() if agent_num is None else agent_num
        # agent_num = agent_num if agent_num is not None else td["agent_num"][0].item()

        depot = td["depot"]
        depot = depot.repeat(1, agent_num + 1, 1)
        loc = td["locs"]
        left_request = loc.size(1) // 2
        whole_instance = torch.cat((depot, loc), dim=1)

        # Distance from all nodes between each other
        distance = torch.cdist(whole_instance, whole_instance, p=2)
        index = torch.arange(left_request, 2 * left_request, device=depot.device)[
            None, :, None
        ]
        index = index.repeat(distance.shape[0], 1, 1)
        add_pd_distance = distance[
            :, agent_num + 1 : agent_num + 1 + left_request, agent_num + 1 :
        ].gather(-1, index)
        add_pd_distance = add_pd_distance.squeeze(-1)

        remain_pickup_max_distance = distance[:, 0, : agent_num + 1 + left_request].max(
            dim=-1, keepdim=True
        )[0]
        remain_delivery_max_distance = distance[:, 0, agent_num + 1 + left_request :].max(
            dim=-1, keepdim=True
        )[0]
        remain_sum_paired_distance = add_pd_distance.sum(dim=-1, keepdim=True)

        # Distance from depot to all nodes
        # Delivery nodes should consider the sum of distance from depot to paired pickup nodes and pickup nodes to delivery nodes
        distance[:, 0, agent_num + 1 : agent_num + 1 + left_request] = (
            distance[:, 0, agent_num + 1 : agent_num + 1 + left_request]
            + distance[:, 0, agent_num + 1 + left_request :]
        )

        # Distance from depot to all nodes
        depot_distance = distance[:, 0, :]
        depot_distance[:, agent_num + 1 : agent_num + 1 + left_request] = depot_distance[
            :, agent_num + 1 : agent_num + 1 + left_request
        ]  # + add_pd_distance

        batch_size, n_loc, _ = loc.size()
        to_delivery = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    n_loc // 2 + agent_num + 1,
                    dtype=torch.uint8,
                    device=loc.device,
                ),
                torch.zeros(
                    batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device
                ),
            ],
            dim=-1,
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((depot, loc), -2),
                "visited": torch.zeros(
                    batch_size,
                    1,
                    n_loc + agent_num + 1,
                    dtype=torch.uint8,
                    device=loc.device,
                ),
                "lengths": torch.zeros(batch_size, agent_num, device=loc.device),
                "longest_lengths": torch.zeros(batch_size, agent_num, device=loc.device),
                "cur_coord": td["depot"]
                if len(td["depot"].shape) == 2
                else td["depot"].squeeze(1),
                "i": torch.zeros(
                    batch_size, dtype=torch.int64, device=loc.device
                ),  # Vector with length num_steps
                "to_delivery": to_delivery,
                "count_depot": torch.zeros(
                    batch_size, 1, dtype=torch.int64, device=loc.device
                ),
                "agent_idx": torch.ones(
                    batch_size, 1, dtype=torch.long, device=loc.device
                ),
                "left_request": left_request
                * torch.ones(batch_size, 1, dtype=torch.long, device=loc.device),
                "remain_pickup_max_distance": remain_pickup_max_distance,
                "remain_delivery_max_distance": remain_delivery_max_distance,
                "depot_distance": depot_distance,
                "remain_sum_paired_distance": remain_sum_paired_distance,
                "add_pd_distance": add_pd_distance,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Get the action mask for the current state."""

        visited_loc = td["visited"].clone()

        agent_num = td["lengths"].size(1)
        n_loc = visited_loc.size(-1) - agent_num - 1  # num of customers
        batch_size = visited_loc.size(0)
        agent_idx = td["agent_idx"][:, None, :]
        mask_loc = visited_loc.to(td["to_delivery"].device) | (1 - td["to_delivery"])

        # depot
        if td["i"][0].item() != 0:
            mask_loc[:, :, : agent_num + 1] = 1

            # if deliver nodes which is assigned agent is complete, then agent can go to depot
            no_item_to_delivery = (
                visited_loc[:, :, n_loc // 2 + agent_num + 1 :]
                == td["to_delivery"][:, :, n_loc // 2 + agent_num + 1 :]
            ).all(dim=-1)
            mask_loc[no_item_to_delivery.squeeze(-1), :, :] = mask_loc[
                no_item_to_delivery.squeeze(-1), :, :
            ].scatter_(-1, agent_idx[no_item_to_delivery.squeeze(-1), :, :], 0)

            condition = (td["count_depot"] == agent_num - 1) & (
                (visited_loc[:, :, agent_num + 1 :] == 0).sum(dim=-1) != 0
            )

            mask_loc[..., agent_num][condition] = 1

        else:
            return (
                torch.cat(
                    [
                        torch.zeros(
                            batch_size, 1, 1, dtype=torch.uint8, device=mask_loc.device
                        ),
                        torch.ones(
                            batch_size,
                            1,
                            n_loc + agent_num,
                            dtype=torch.uint8,
                            device=mask_loc.device,
                        ),
                    ],
                    dim=-1,
                )
                > 0
            )
        action_mask = mask_loc == 0  # action_mask gets feasible actions
        return action_mask

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Calculate the reward (negative tour length)
        if self.objective == "minmax":
            return -td["lengths"].max(dim=-1, keepdim=True)[0].squeeze(-1)
        elif self.objective == "minsum":
            return -td["lengths"].sum(dim=-1, keepdim=True).squeeze(-1)
        else:
            raise ValueError(f"Unknown objective {self.objective}")

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        assert True, "Not implemented"

    def generate_data(self, batch_size) -> TensorDict:
        # Batch size input check
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

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
        max_nodes = self.num_loc + self.max_num_agents + 1
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(max_nodes, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(max_nodes, 1),
                dtype=torch.bool,
            ),
            visited=UnboundedDiscreteTensorSpec(
                shape=(1, max_nodes),
                dtype=torch.bool,
            ),
            lengths=UnboundedContinuousTensorSpec(
                shape=(self.max_num_agents,),
                dtype=torch.float32,
            ),
            longest_lengths=UnboundedContinuousTensorSpec(
                shape=(self.max_num_agents,),
                dtype=torch.float32,
            ),
            cur_coord=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(2,),
                dtype=torch.float32,
            ),
            to_delivery=UnboundedDiscreteTensorSpec(
                shape=(max_nodes, 1),
                dtype=torch.bool,
            ),
            count_depot=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            agent_idx=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            left_request=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            remain_pickup_max_distance=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            remain_delivery_max_distance=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            depot_distance=UnboundedContinuousTensorSpec(
                shape=(max_nodes,),
                dtype=torch.float32,
            ),
            remain_sum_paired_distance=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            add_pd_distance=UnboundedContinuousTensorSpec(
                shape=(max_nodes,),
                dtype=torch.float32,
            ),
            ## NOTE: we should have a vectorized implementation for agent_num
            # agent_num=UnboundedDiscreteTensorSpec(
            #     shape=(1,),
            #     dtype=torch.int64,
            # ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=max_nodes,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        # TODO: color switch with new agents; add pickup and delivery nodes as in `PDPEnv.render`

        import matplotlib.pyplot as plt
        import numpy as np

        from matplotlib import cm, colormaps

        num_routine = (actions == 0).sum().item() + 2
        base = colormaps["nipy_spectral"]
        color_list = base(np.linspace(0, 1, num_routine))
        cmap_name = base.name + str(num_routine)
        out = base.from_list(cmap_name, color_list, num_routine)

        if ax is None:
            # Create a plot of the nodes
            _, ax = plt.subplots()

        td = td.detach().cpu()

        if actions is None:
            actions = td.get("action", None)

        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]

        locs = td["locs"]

        # add the depot at the first action and the end action
        actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

        # gather locs in order of action if available
        if actions is None:
            log.warning("No action in TensorDict, rendering unsorted locs")
        else:
            locs = locs

        # Cat the first node to the end to complete the tour
        x, y = locs[:, 0], locs[:, 1]

        # plot depot
        ax.scatter(
            locs[0, 0],
            locs[0, 1],
            edgecolors=cm.Set2(2),
            facecolors="none",
            s=100,
            linewidths=2,
            marker="s",
            alpha=1,
        )

        # plot visited nodes
        ax.scatter(
            x[1:],
            y[1:],
            edgecolors=cm.Set2(0),
            facecolors="none",
            s=50,
            linewidths=2,
            marker="o",
            alpha=1,
        )

        # text depot
        ax.text(
            locs[0, 0],
            locs[0, 1] - 0.025,
            "Depot",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color=cm.Set2(2),
        )

        # plot actions
        color_idx = 0
        for action_idx in range(len(actions) - 1):
            if actions[action_idx] == 0:
                color_idx += 1
            from_loc = locs[actions[action_idx]]
            to_loc = locs[actions[action_idx + 1]]
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color=out(color_idx),
                lw=1,
            )
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="-|>", color=out(color_idx)),
                size=15,
                annotation_clip=False,
            )

        # Setup limits and show
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        plt.show()
