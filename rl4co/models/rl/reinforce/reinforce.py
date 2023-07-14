from typing import Any, Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline, get_reinforce_baseline
from rl4co.utils.lightning import get_lightning_device
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCE(RL4COLitModule):
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
        baseline_kwargs={},
        **kwargs,
    ):
        super().__init__(env, policy, **kwargs)

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        extra = td.get("extra", None)

        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out: dict = self.policy(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            bl_val, bl_neg_loss = (
                self.baseline.eval(td, "train", extra)
                if self.baseline is not None
                else (extra, 0)
            )

            advantage = out["reward"] - bl_val  # advantage = reward - baseline
            reinforce_loss = -(advantage * out["log_likelihood"]).mean()
            loss = reinforce_loss - bl_neg_loss
            out.update(
                {
                    "loss": loss,
                    "reinforce_loss": reinforce_loss,
                    "bl_loss": -bl_neg_loss,
                    "bl_val": bl_val,
                }
            )

        metrics = self.log_metrics(out, "train")
        return {"loss": out.get("loss", None), **metrics}

    def post_setup_hook(self, stage="fit"):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.data_cfg["val_size"],
        )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.self.data_cfg["val_size"],
        )

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
        )
