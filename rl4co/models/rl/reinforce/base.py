from tensordict import TensorDict
from torch import nn

from rl4co.utils.lightning import get_lightning_device


class REINFORCE(nn.Module):
    """Base model for REINFORCE-based models

    Args:
        env: TorchRL Environment
        policy: Policy (set up in model)
        baseline: REINFORCE Baseline (set up in model)
    """

    def __init__(self, env, policy=None, baseline=None):
        super(REINFORCE, self).__init__()
        self.env = env

    def forward(self, td: TensorDict, phase: str = "train", extra=None, **policy_kwargs):
        # Evaluate model, get costs and log probabilities
        out = self.policy(td, phase, **policy_kwargs)

        if phase == "train":
            # REINFORCE loss: we consider the rewards instead of costs to be consistent with the literature
            bl_val, bl_neg_loss = (
                self.baseline.eval(td, out["reward"]) if extra is None else (extra, 0)
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

        return out

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
