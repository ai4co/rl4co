import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from rl4co.models.zoo.ptrnet.policy import PointerNetworkPolicy
from rl4co.models.rl.reinforce import WarmupBaseline, RolloutBaseline
from rl4co.utils.lightning import get_lightning_device


class PointerNetwork(nn.Module):
    def __init__(self, env, policy=None, baseline=None):
        """
        Pointer Network for neural combinatorial optimization based on REINFORCE
        Based on Vinyals et al. (2015) https://arxiv.org/abs/1506.03134
        Refactored from reference implementation: https://github.com/wouterkool/attention-learn-to-route

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
        """
        super().__init__()
        self.env = env
        self.policy = PointerNetworkPolicy(env) if policy is None else policy
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )

    def forward(
        self, td: TensorDict, phase: str = "train", decode_type: str = None
    ) -> TensorDict:
        """Evaluate model, get costs and log probabilities and compare with baseline"""

        # Evaluate modelim=0, get costs and log probabilities
        out = self.policy(td)

        if phase == "train":
            cost = -out["reward"]
            ll = out["log_likelihood"]

            # Calculate loss
            bl_val, bl_loss = self.baseline.eval(td, cost)

            advantage = cost - bl_val
            reinforce_loss = (advantage * ll).mean()
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
        if hasattr(self.baseline, "setup"):
            self.baseline.setup(
                self.policy,
                lit_module.train_dataloader(),
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
