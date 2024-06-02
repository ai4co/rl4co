from typing import Any, Optional, Union

import math
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline, get_reinforce_baseline
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TB(REINFORCE):
    """REINFORCE algorithm, also known as policy gradients.
    See superclass `RL4COLitModule` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string
        **kwargs: Keyword arguments passed to the superclass
    """
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: Union[REINFORCEBaseline, str] = "no",
        baseline_kwargs: dict = {},
        beta: int = 1,
        gfn_epochs: int = 2,
        **kwargs,
    ):
        super().__init__(env, policy, **kwargs)
        self.save_hyperparameters(logger=False)

        if baseline == "critic":
            log.warning(
                "Using critic as baseline. If you want more granular support, use the A2C module instead."
            )

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline
        
        self.gfn_cfg = {
            "beta": beta,
            "epochs": gfn_epochs,
        }
        
    # def shared_step(
    #     self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    # ):
    #     td = self.env.reset(batch)
    #     # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
    #     out = self.policy(td, self.env, phase=phase)

    #     # Compute loss
    #     if phase == "train":
    #         out = self.calculate_loss(td, batch, out)

    #     metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
    #     return {"loss": out.get("loss", None), **metrics}

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        
        forward_flow = log_likelihood + policy_out["log_z"].view(-1)
        backward_flow = self.gfn_cfg["beta"] * (reward - bl_val) + math.log(1/(2*td["locs"].size(1)))
        # import pdb; pdb.set_trace()
        
        tb_loss = torch.pow(forward_flow-backward_flow, 2).mean()
        loss = tb_loss + bl_loss
        
        policy_out.update(
            {
                "loss": loss,
                "tb_loss": tb_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out


class TB_offline(RL4COLitModule):
    """REINFORCE algorithm, also known as policy gradients.
    See superclass `RL4COLitModule` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string
        **kwargs: Keyword arguments passed to the superclass
    """
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs: dict = {},
        beta: int = 1,
        gfn_epochs: int = 2,
        mini_batch_size: Union[int, float] = 0.25,
        **kwargs,
    ):
        super().__init__(env, policy, **kwargs)
        self.automatic_optimization = False  # manual optimization
        self.save_hyperparameters(logger=False)

        if baseline == "critic":
            log.warning(
                "Using critic as baseline. If you want more granular support, use the A2C module instead."
            )

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline
        
        self.gfn_cfg = {
            "beta": beta,
            "epochs": gfn_epochs,
            "mini_batch_size": mini_batch_size,
        }
        
    def on_train_epoch_end(self):
        """
        ToDo: Add support for other schedulers.
        """

        sch = self.lr_schedulers()

        # If the selected scheduler is a MultiStepLR scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.MultiStepLR):
            sch.step()
        
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
       # Evaluate old actions, log probabilities, and rewards
        with torch.no_grad():
            td = self.env.reset(batch)  # note: clone needed for dataloader
            out = self.policy(td.clone(), self.env, phase=phase, return_actions=True)

        if phase == "train":
            batch_size = out["actions"].shape[0]

            # infer batch size
            if isinstance(self.gfn_cfg["mini_batch_size"], float):
                mini_batch_size = int(batch_size * self.gfn_cfg["mini_batch_size"])
            elif isinstance(self.gfn_cfg["mini_batch_size"], int):
                mini_batch_size = self.gfn_cfg["mini_batch_size"]
            else:
                raise ValueError("mini_batch_size must be an integer or a float.")

            if mini_batch_size > batch_size:
                mini_batch_size = batch_size

            # Todo: Add support for multi dimensional batches
            td.set("logprobs", out["log_likelihood"])
            td.set("reward", out["reward"])
            td.set("action", out["actions"])

            # Inherit the dataset class from the environment for efficiency
            dataset = self.env.dataset_cls(td)
            dataloader = DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=True,
                collate_fn=dataset.collate_fn,
            )

            for _ in range(self.gfn_cfg["epochs"]):  # GFN inner epoch, K
                for sub_td in dataloader:
                    sub_td = sub_td.to(td.device)
                    previous_reward = sub_td["reward"].view(-1, 1)
                    out = self.policy(  # note: remember to clone to avoid in-place replacements!
                        sub_td.clone(),
                        actions=sub_td["action"],
                        env=self.env,
                    )

                    forward_flow = out["log_likelihood"] + out["log_z"].view(-1)
                    backward_flow = self.gfn_cfg["beta"] * out["reward"] + math.log(1/(2*sub_td["locs"].size(1)))  # MaxEnt Pb
                    # import pdb; pdb.set_trace()
                    
                    tb_loss = torch.pow(forward_flow-backward_flow, 2).mean()
                    loss = tb_loss #+ bl_loss
                    
                    # compute total loss
                    # loss = (
                    #     surrogate_loss
                    #     + self.ppo_cfg["vf_lambda"] * value_loss
                    #     - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                    # )

                    # perform manual optimization following the Lightning routine
                    # https://lightning.ai/docs/pytorch/stable/common/optimization.html

                    opt = self.optimizers()
                    opt.zero_grad()
                    self.manual_backward(loss)
                    opt.step()

            out.update(
                {
                    "loss": loss,
                }
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}