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
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.trainer import RL4COTrainer
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SVRPEnv(RL4COEnvBase):
    """
    Basic Skill-VRP environment.
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
            # TODO should these be integer or float?
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
        print("current_node:\n", td["current_node"])
        print("visited:\n", td["visited"].squeeze(-1))
        print("action_mask:\n", td["action_mask"])
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
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
                "tech_costs": torch.tensor(self.tech_costs, device=self.device),
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
        print("Calculate reward...")
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

        return (distances * costs).sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        print("check_solution_validity...")
        # TODO
        None

    @staticmethod
    def render(td: TensorDict, actions=None, **kwargs):
        # TODO
        None

    @staticmethod
    def load_data(fpath, batch_size=...):
        print("load_data...")
        # TODO
        return None


if __name__ == "__main__":
    import argparse
    from rl4co.models.nn.utils import rollout, random_policy

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", help="Number of epochs to train for", type=int, default=3
    )
    args = parser.parse_args()

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

    env = SVRPEnv(num_loc=20, device=device_str)
    td_init = env.reset(batch_size=[batch_size]).to(device)

    ### --- random policy --- ###
    reward, td, actions = rollout(
        env=env,
        td=td_init.clone(),
        policy=random_policy,
        max_steps=1000,
    )
    print("reward", reward)
    print("actions", actions)
    print("td", td)

    assert reward.shape == (batch_size,)

    ### --- AM --- ###
    # Model: default is AM with REINFORCE and greedy rollout baseline
    model = AttentionModel(
        env=env,
        baseline="rollout",
        train_data_size=100_000,
        val_data_size=10_000,
    )

    # Greedy rollouts over untrained model
    model = model.to(device)
    out = model(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

    print("out", out)

    ### --- Logging --- ###
    from datetime import date, datetime
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    date_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    wandb.login()
    logger = WandbLogger(
        project="routefinder",
        name=f"skill-vrp_am_{date_time_str}",
    )

    ### --- Training --- ###
    # The RL4CO trainer is a wrapper around PyTorch Lightning's `Trainer` class which adds some functionality and more efficient defaults
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
    )

    # fit model
    trainer.fit(model)

    ### --- Testing --- ###
    trainer.test(model)
