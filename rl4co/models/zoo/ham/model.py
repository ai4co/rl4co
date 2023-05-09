import torch
from torch import nn
from tensordict import TensorDict
import lightning as L

from rl4co.models.rl.reinforce import WarmupBaseline, RolloutBaseline
from rl4co.utils.lightning import get_lightning_device
from rl4co.models.zoo.ham.policy import HeterogeneousAttentionModelPolicy


class HeterogeneousAttentionModel(nn.Module):
    def __init__(self, env, policy=None, baseline=None):
        """
        Heterogenous Attention Model for solving the Pickup and Delivery Problem based on REINFORCE
        https://arxiv.org/abs/2110.02634

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
        """
        super().__init__()
        self.env = env
        self.policy = HeterogeneousAttentionModelPolicy(env) if policy is None else policy
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )

    def forward(self, td: TensorDict, phase: str = "train", **policy_kwargs):
        # Evaluate model, get costs and log probabilities
        out = self.policy(td, phase, **policy_kwargs)

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
