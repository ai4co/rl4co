import torch
from torch import nn
from tensordict import TensorDict
import lightning as L

from rl4co.utils.ops import unbatchify
from rl4co.models.zoo.pomo.utils import get_best_actions
from rl4co.models.zoo.pomo.policy import POMOPolicy
from rl4co.models.zoo.pomo.augmentations import StateAugmentation
from rl4co.models.rl.reinforce.baselines import WarmupBaseline, RolloutBaseline
from rl4co.models.rl.reinforce.base import REINFORCE


class POMO(REINFORCE):
    """
    POMO Model for neural combinatorial optimization based on REINFORCE
    Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011

    Args:
        env: TorchRL Environment
        policy: Policy
        baseline: REINFORCE Baseline
        num_augment: Number of augmentations (default: 8)
    """
    def __init__(self, env, policy=None, baseline=None, num_augment=8):
        super().__init__(env, policy, baseline)
        self.policy = POMOPolicy(self.env) if policy is None else policy

        # TODO: check baseline
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )
        # POMO parameters
        self.num_pomo = self.policy.num_pomo
        self.num_augment = num_augment
        self.augment = (
            StateAugmentation(self.env.name, num_augment) if num_augment > 1 else None
        )

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        decode_type: str = "sampling",
        return_actions: bool = False,
    ):
        """Evaluate model, get costs and log probabilities and compare with baseline"""

        # Augment data if not in training phase
        if phase != "train" and self.augment is not None:
            td = self.augment(td)

        # Evaluate model, get costs and log probabilities
        out = self.policy(td, decode_type=decode_type, return_actions=return_actions)

        # Max POMO reward. Decouple augmentation and POMO
        # [batch, num_pomo, num_augment]
        reward = unbatchify(
            unbatchify(out["reward"], self.num_augment if phase != "train" else 1),
            self.num_pomo,
        )
        max_reward, max_idxs = reward.max(dim=1)
        out.update(
            {
                "max_reward": max_reward,
                "best_actions": get_best_actions(out["actions"], max_idxs)
                if return_actions
                else None,
            }
        )

        if phase == "train":
            costs = unbatchify(-out["reward"], self.policy.num_pomo)
            ll = unbatchify(out["log_likelihood"], self.policy.num_pomo)
            bl_val, bl_loss = self.baseline.eval(td, costs)

            # Calculate REINFORCE loss
            advantage = costs - bl_val
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

        # Get augmentation score only during inference
        if phase != "train" and self.augment is not None:
            # [batch, num_augment]
            aug_reward = unbatchify(max_reward, self.num_augment)
            max_aug_reward, max_idxs = aug_reward.max(dim=1)
            out.update(
                {
                    "max_aug_reward": max_aug_reward,
                    "best_aug_actions": get_best_actions(out["actions"], max_idxs)
                    if return_actions
                    else None,
                }
            )

        return out