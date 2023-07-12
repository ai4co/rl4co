from typing import Any, Union

from rl4co.algos.common.base import RL4COLitModule
from rl4co.algos.reinforce.baselines import (
    REINFORCE_BASELINES_REGISTRY,
    REINFORCEBaseline,
)
from rl4co.utils.lightning import get_lightning_device
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCE(RL4COLitModule):
    """REINFORCE algorithm, also known as policy gradients.


    Args:
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string.
    """

    def __init__(
        self,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs={},
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(baseline, str):
            baseline_cls = REINFORCE_BASELINES_REGISTRY.get(baseline, None)
            if baseline_cls is None:
                raise ValueError(
                    f"Unknown baseline {baseline_cls}. Available baselines: {REINFORCE_BASELINES_REGISTRY.keys()}"
                )
            self.baseline = baseline_cls(**baseline_kwargs)

        else:
            self.baseline = baseline
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        extra = td.get("extra", None)

        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out: dict = self.policy(td, "train", extra)

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

    def setup(self):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.cfg.data.val_size,
        )

    def on_train_epoch_end(self):
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.cfg.data.val_size,
        )

    def wrap_dataset(self, dataset):
        """Wrap dataset for baseline evaluation"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
        )
