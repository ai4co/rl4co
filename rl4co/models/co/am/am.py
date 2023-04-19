import torch
from torch import nn
from tensordict import TensorDict
import lightning as L

from rl4co.models.co.am.policy import AttentionModelPolicy
from rl4co.models.rl.reinforce import WarmupBaseline, RolloutBaseline
from rl4co.utils.lightning import get_lightning_device


class AttentionModel(nn.Module):
    def __init__(self, env, policy=None, baseline=None):
        """
        Attention Model for neural combinatorial optimization based on REINFORCE
        Based on Wouter Kool et al. (2018) https://arxiv.org/abs/1803.08475
        Refactored from reference implementation: https://github.com/wouterkool/attention-learn-to-route

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
        """
        super().__init__()
        self.env = env
        self.policy = AttentionModelPolicy(env) if policy is None else policy
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )

    def forward(
        self, td: TensorDict, phase: str = "train", decode_type: str = None
    ) -> TensorDict:
        # Evaluate model, get costs and log probabilities
        out = self.policy(td)

        if phase == "train":
            # Evaluate baseline
            bl_val, bl_loss = self.baseline.eval(td, -out["reward"])

            # Calculate loss
            advantage = -out["reward"] - bl_val
            reinforce_loss = (advantage * out["log_likelihood"]).mean()
            loss = reinforce_loss + bl_loss
            out.update(
                {
                    "loss": loss,
                    "reinforce_loss": reinforce_loss,
                    "bl_loss": bl_loss,
                    "bl_val": bl_val,
                }
            )

        return out

    def setup(self, lit_module):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            lit_module.val_dataloader(),
            self.env,
            device=get_lightning_device(lit_module),
        )

    def on_train_epoch_end(self, lit_module):
        self.baseline.epoch_callback(
            self.policy,
            lit_module.val_dataloader(),
            lit_module.current_epoch,
            self.env,
            device=get_lightning_device(lit_module),
        )
