import torch
from torch import nn
from tensordict import TensorDict
import lightning as L

from ncobench.utils.lightning import get_lightning_device
from ncobench.utils.ops import undo_repeat_batch
from ncobench.models.co.am.policy import AttentionModelPolicy
from ncobench.models.rl.reinforce import WarmupBaseline, RolloutBaseline
from ncobench.models.co.pomo.utils import get_best_actions


class POMO(nn.Module):
    def __init__(self, env, policy=None, baseline=None, num_augment=8):
        """
        POMO Model for neural combinatorial optimization based on REINFORCE
        Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
            num_augment: Number of augmentations (default: 8)
        """
        super().__init__()
        self.env = env
        self.policy = AttentionModelPolicy(env) if policy is None else policy
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )

    def forward(self, td: TensorDict, phase: str="train", decode_type: str="sampling", return_actions: bool=False) -> TensorDict:
        """Evaluate model, get costs and log probabilities and compare with baseline"""

        # Augment data if not in training phase
        if phase != "train" and self.augment is not None:
            td = self.augment(td)

        # Evaluate model, get costs and log probabilities
        out = self.policy(td, decode_type=decode_type, return_actions=return_actions)

        costs = undo_repeat_batch(-out['reward'], self.policy.num_pomo)
        ll = undo_repeat_batch(out['log_likelihood'], self.policy.num_pomo)
        bl_val, bl_loss = self.baseline.eval(td, costs)

        # Calculate REINFORCE loss
        advantage = costs - bl_val
        reinforce_loss = (advantage * ll).mean()
        loss = reinforce_loss + bl_loss

        # Max POMO reward. Decouple augmentation and POMO 
        # [num_pomo, num_augment, batch]
        reward = undo_repeat_batch(undo_repeat_batch(out["reward"], self.num_augment if phase != "train" else 1), self.num_pomo, dim=1)
        max_reward, max_idxs = reward.max(dim=0)
        pomo_retvals = {"max_reward": max_reward, "best_actions": get_best_actions(out["actions"], max_idxs) if return_actions else None}

        # Get augmentation score only during inference
        aug_retvals = {}
        if phase != "train" and self.augment is not None:
            # [num_augment, batch]
            aug_reward = undo_repeat_batch(max_reward, self.num_augment)
            max_aug_reward, max_idxs = aug_reward.max(dim=0)
            aug_retvals = {"max_aug_reward": max_aug_reward, "best_aug_actions": get_best_actions(out["actions"], max_idxs) if return_actions else None}
 
        return {'loss': loss, 'reinforce_loss': reinforce_loss, 'bl_loss': bl_loss, 'bl_val': bl_val, **out, **pomo_retvals, **aug_retvals}

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
