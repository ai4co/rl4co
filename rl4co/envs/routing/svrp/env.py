from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SVRPEnv(RL4COEnvBase):
    """
    Basic Skill-VRP environment. The environment is a variant of the Capacitated Vehicle Routing Problem (CVRP).
    Each technician has a certain skill-level and each customer node requires a certain skill-level to be serviced.
    Each customer node needs is to be serviced by exactly one technician. Technicians can only service nodes if
    their skill-level is greater or equal to the required skill-level of the node. The environment is episodic and
    the goal is to minimize the total travel cost of the technicians. The travel cost depends on the skill-level of
    the technician. The environment is defined by the following parameters:

    Args:
        num_loc (int): Number of customer locations. Default: 20
        min_loc (float): Minimum value for the location coordinates. Default: 0
        max_loc (float): Maximum value for the location coordinates. Default: 1
        min_skill (float): Minimum skill level of the technicians. Default: 1
        max_skill (float): Maximum skill level of the technicians. Default: 10
        tech_costs (list): List of travel costs for the technicians. Default: [1, 2, 3]. The number of entries in this list determines the number of available technicians.
        td_params (TensorDict): Parameters for the TensorDict. Default: None
    """

    name = "svrp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_skill: float = 1,
        max_skill: float = 10,
        tech_costs: list = [1, 2, 3],
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_skill = min_skill
        self.max_skill = max_skill
        self.tech_costs = tech_costs
        self.num_tech = len(tech_costs)
        self._make_spec(td_params)

    def _make_spec(self, td_params: TensorDict = None):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=self.min_loc,
                high=self.max_loc,
                shape=(self.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            skills=BoundedTensorSpec(
                low=self.min_skill,
                high=self.max_skill,
                shape=(self.num_loc, 1),
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=self.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32)
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def generate_data(self, batch_size):
        """Generate data for the basic Skill-VRP. The data consists of the locations of the customers,
        the skill-levels of the technicians and the required skill-levels of the customers.
        The data is generated randomly within the given bounds."""
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize technicians and sort ascendingly
        techs, _ = torch.sort(
            torch.FloatTensor(*batch_size, self.num_tech, 1)
            .uniform_(self.min_skill, self.max_skill)
            .to(self.device),
            dim=-2,
        )

        # Initialize the skills
        skills = (
            torch.FloatTensor(*batch_size, self.num_loc, 1).uniform_(0, 1).to(self.device)
        )
        # scale skills
        skills = torch.max(techs, dim=1, keepdim=True).values * skills
        td = TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "techs": techs,
                "skills": skills,
            },
            batch_size=batch_size,
            device=self.device,
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Calculates the action mask for the Skill-VRP. The action mask is a binary mask that indicates which customer nodes can be services, given the previous decisions.
        For the Skill-VRP, a node can be serviced if the technician has the required skill-level and the node has not been visited yet.
        The depot cannot be visited if there are still unserved nodes and the technician either just visited the depot or is the last technician
        (because every customer node needs to be visited).
        """
        batch_size = td["locs"].shape[0]
        # check skill level
        current_tech_skill = gather_by_index(td["techs"], td["current_tech"]).reshape(
            [batch_size, 1]
        )
        can_service = td["skills"] <= current_tech_skill.unsqueeze(1).expand_as(
            td["skills"]
        )
        mask_loc = td["visited"][..., 1:, :].to(can_service.dtype) | ~can_service
        # Cannot visit the depot if there are still unserved nodes and I either just visited the depot or am the last technician
        mask_depot = (
            (td["current_node"] == 0) | (td["current_tech"] == td["techs"].size(-2) - 1)
        ) & ((mask_loc == 0).int().sum(-2) > 0)
        return ~torch.cat((mask_depot[..., None], mask_loc), -2).squeeze(-1)

    def _step(self, td: TensorDict) -> torch.Tensor:
        """Step function for the Skill-VRP. If a technician returns to the depot, the next technician is sent out.
        The visited node is marked as visited. The reward is set to zero and the done flag is set if all nodes have been visited.
        """
        current_node = td["action"][:, None]  # Add dimension for step

        # if I go back to the depot, send out next technician
        td["current_tech"] += (current_node == 0).int()

        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-2, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-2) == visited.size(-2)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[0]
        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        self.to(td.device)

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "techs": td["techs"],
                "skills": td["skills"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "current_tech": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=self.device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + 1, 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Calculated the reward, where the reward is the negative total travel cost of the technicians.
        The travel cost depends on the skill-level of the technician."""
        # Check that the solution is valid
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather dataset in order of tour
        batch_size = td["locs"].shape[0]
        depot = td["locs"][..., 0:1, :]
        locs_ordered = torch.cat(
            [
                depot,
                gather_by_index(td["locs"], actions).reshape(
                    [batch_size, actions.size(-1), 2]
                ),
            ],
            dim=1,
        )

        # calculate travelling costs depending on the technicians' skill level
        costs = torch.zeros(batch_size, locs_ordered.size(-2), device=self.device)
        indices = torch.nonzero(actions == 0)
        start = tech = 0
        batch = 0
        for each in indices:
            if each[0] > batch:
                costs[batch, start:] = self.tech_costs[tech]
                start = tech = 0
                batch = each[0]
            end = (
                each[-1] + 1
            )  # indices in locs_ordered are shifted by one due to added depot in the front
            costs[batch, start:end] = self.tech_costs[tech]
            tech += 1
            start = end
        costs[batch, start:] = self.tech_costs[tech]

        travel_to = torch.roll(locs_ordered, -1, dims=-2)
        distances = get_distance(locs_ordered, travel_to)
        return -(distances * costs).sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are not visited twice except depot and required skill levels are always met."""
        batch_size, graph_size = td["skills"].shape[0], td["skills"].shape[1]
        sorted_pi = actions.data.sort(1).values

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, graph_size)
            == sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # make sure all required skill  levels are met
        indices = torch.nonzero(actions == 0)
        skills = torch.cat(
            [torch.zeros(batch_size, 1, 1, device=td.device), td["skills"]], 1
        )
        skills_ordered = gather_by_index(skills, actions).reshape(
            [batch_size, actions.size(-1), 1]
        )
        batch = start = tech = 0
        for each in indices:
            if each[0] > batch:
                start = tech = 0
                batch = each[0]
            assert (
                skills_ordered[batch, start : each[1]] <= td["techs"][batch, tech]
            ).all(), "Skill level not met"
            start = each[1] + 1  # skip the depot
            tech += 1

    @staticmethod
    def render(
        td: TensorDict,
        actions=None,
        ax=None,
        **kwargs,
    ):
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
        plt.show()
