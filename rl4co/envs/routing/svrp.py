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
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SVRPEnv(RL4COEnvBase):
    """
    Basic Skill-VRP environment.
    """

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
        print("SVRPEnv init...")
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
        print("_make_spec...")
        # TODO
        None

    def generate_data(self, batch_size):
        print("generate_data...")
        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize technicians
        techs = (
            torch.FloatTensor(*batch_size, self.num_tech, 1)
            .uniform_(self.min_skill, self.max_skill)
            .to(self.device)
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
        print("td: ", td)
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        print("get_action_mask...")
        # check skill level
        current_tech_skill = gather_by_index(td["techs"], td["current_tech"]).reshape(
            [batch_size, 1]
        )
        can_service = td["skills"] <= current_tech_skill.unsqueeze(1).expand_as(
            td["skills"]
        )
        mask_loc = td["visited"][..., 1:].to(can_service.dtype) | ~can_service
        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (td["current_node"] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        print(
            "action_mask:", ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)
        )
        return ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)

    def _step(self, td: TensorDict) -> torch.Tensor:
        print("_step...")
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["skills"].size(-1)  # Excludes depot

        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)

        # if I go back to the depot, set to next technician
        # TODO

        # if last technician, set done to True??
        # TODO

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
        print("_reset...")
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]
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
                    (*batch_size, 1, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        print("td_reset: ", td_reset)
        return td_reset

    def get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Check that the solution is valid
        # if self.check_solution:
        #     self.check_solution_validity(td, actions)

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

        return None

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        print("check_solution_validity...")
        None

    @staticmethod
    def render(td: TensorDict, actions=None, **kwargs):
        None

    @staticmethod
    def load_data(fpath, batch_size=...):
        print("load_data...")
        return None


if __name__ == "__main__":
    from rl4co.models.nn.utils import rollout, random_policy

    # device
    device_str = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
            else "cpu"
        )
    )
    device = torch.device(device_str)
    # env
    batch_size = 3
    env = SVRPEnv(device=device_str)
    reward, td, actions = rollout(
        env=env,
        td=env.reset(batch_size=[batch_size]).to(device),
        policy=random_policy,
        max_steps=1000,
    )
