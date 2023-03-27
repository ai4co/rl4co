from torch import nn
from tensordict import TensorDict


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
        bl_val, bl_loss = self.baseline.eval(td, out_policy["cost"])

        # print(bl_val, bl_loss)
        # Calculate loss
        advantage = out_policy["cost"] - bl_val
        reinforce_loss = (advantage * out_policy["log_likelihood"]).mean()
        loss = reinforce_loss + bl_loss

        return {
            "loss": loss,
            "reinforce_loss": reinforce_loss,
            "bl_loss": bl_loss,
            "bl_val": bl_val,
            **out_policy,
        }

    def setup(self, pl_module):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(self.policy, pl_module.val_dataloader(), self.env)

    def on_train_epoch_end(self, pl_module):
        self.baseline.epoch_callback(
            self.policy, pl_module.val_dataloader(), pl_module.current_epoch, self.env
        )
