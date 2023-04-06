import torch
from torch import nn
from tensordict import TensorDict
import lightning as L

from ncobench.utils.lightning import get_lightning_device


class AttentionModel(nn.Module):
    def __init__(self, env, policy, baseline):
        """
        Attention Model for neural combinatorial optimization
        Based on Wouter Kool et al. (2018) https://arxiv.org/abs/1803.08475

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
        """
        super().__init__()
        self.env = env
        self.policy = policy
        self.baseline = baseline

    def forward(
        self, td: TensorDict, phase: str = "train", decode_type: str = None
    ) -> TensorDict:
        # Evaluate model, get costs and log probabilities
        out_policy = self.policy(td)
        bl_val, bl_loss = self.baseline.eval(td, -out_policy["reward"])

        # Calculate loss
        advantage = -out_policy["reward"] - bl_val
        reinforce_loss = (advantage * out_policy["log_likelihood"]).mean()
        loss = reinforce_loss + bl_loss

        return {
            "loss": loss,
            "reinforce_loss": reinforce_loss,
            "bl_loss": bl_loss,
            "bl_val": bl_val,
            **out_policy,
        }

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
