from typing import Any

from rl4co.algo.base import RL4COLitModule
from rl4co.utils.lightning import get_lightning_device


class REINFORCE(RL4COLitModule):
    def __init__(self, baseline=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        extra = td.get("extra", None)

        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out: dict = self.policy(td, "train", extra)

        # Compute loss
        if phase == "train":
            bl_val, bl_neg_loss = (
                self.baseline.eval(td, "train", extra) if self.baseline is not None else (extra, 0)
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

    def setup(self, lit_module):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=lit_module.val_batch_size,
            device=get_lightning_device(lit_module),
            dataset_size=lit_module.cfg.data.val_size,
        )

    def on_train_epoch_end(self, lit_module):
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=lit_module.val_batch_size,
            device=get_lightning_device(lit_module),
            epoch=lit_module.current_epoch,
            dataset_size=lit_module.cfg.data.val_size,
        )

    def wrap_dataset(self, lit_module, dataset):
        """Wrap dataset for baseline evaluation"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=lit_module.val_batch_size,
            device=get_lightning_device(lit_module),
        )
