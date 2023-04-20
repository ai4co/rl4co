import torch
from torch import nn
from tensordict import TensorDict
import lightning as L

from rl4co.utils.ops import unbatchify
from rl4co.models.rl.reinforce import NoBaseline
from rl4co.models.zoo.symnco.utils import get_best_actions
from rl4co.models.zoo.symnco.policy import SymNCOPolicy
from rl4co.models.zoo.symnco.augmentations import StateAugmentation
from rl4co.models.zoo.symnco.losses import (
    problem_symmetricity_loss,
    solution_symmetricity_loss,
    invariance_loss,
)


class SymNCO(nn.Module):
    def __init__(
        self,
        env,
        policy,
        baseline=None,
        num_augment=8,
        alpha=0.2,
        beta=1,
        augment_test=True,
        **kwargs
    ):
        """
        SymNCO Model for neural combinatorial optimization based on REINFORCE
        Based on Kim et al. (2022) https://arxiv.org/abs/2205.13209

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
            num_augment: Number of augmentations (default: 8)
            alpha: weight for invariance loss
            beta: weight for solution symmetricity loss
            augment_test: whether to augment data during testing as well
        """
        super().__init__()
        self.env = env
        self.policy = policy
        if baseline is not None:
            print(
                "SymNCO uses baselines in the loss functions, so we do not set the baseline here."
            )
        self.baseline = NoBaseline()  # done in loss function

        # Multi-start parameters from policy, default to 1
        self.num_starts = getattr(policy, "num_starts", 1)
        self.num_augment = num_augment
        assert (
            num_augment > 1
        ), "Number of augmentations must be greater than 1 for SymNCO"
        self.augment = StateAugmentation(env.name, num_augment)
        self.augment_test = augment_test
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta  # weight for solution symmetricity loss

    def forward(self, td: TensorDict, phase: str = "train", **policy_kwargs):
        """Evaluate model, get costs and log probabilities and compare with baseline"""

        # Init vals
        loss_retvals, multi_start_retvals, aug_retvals = {}, {}, {}
        return_action = policy_kwargs.get("return_actions", False)

        # Augment data
        if phase == "train" or self.augment_test:
            td = self.augment(td)
            aug_size = (
                self.num_augment
            )  # reward to [batch_size, num_augment, num_starts]
        else:
            aug_size = 1

        # Evaluate model, get costs and log probabilities and more
        out = self.policy(td, **policy_kwargs)
        reward = unbatchify(unbatchify(out["reward"], self.num_starts), aug_size)

        if phase == "train":
            # [batch_size, num_augment, num_starts]
            ll = unbatchify(
                unbatchify(out["log_likelihood"], self.num_starts), aug_size
            )
            loss_ps = problem_symmetricity_loss(reward, ll)
            loss_ss = solution_symmetricity_loss(reward, ll)
            loss_inv = invariance_loss(out["proj_embeddings"], self.num_augment)
            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv
            loss_retvals = {
                "loss": loss,
                "loss_ss": loss_ss,
                "loss_ps": loss_ps,
                "loss_inv": loss_inv,
            }

        else:
            # Get best actions for multi-start # [batch_size, num_augment, num_starts]
            max_reward, max_idxs = reward.max(dim=2)
            multi_start_retvals = {
                "max_reward": max_reward,
                "best_actions": get_best_actions(out["actions"], max_idxs)
                if return_action
                else None,
            }
            # Get best out of augmented # [batch, num_augment]
            max_aug_reward, max_idxs = max_reward.max(dim=1)
            aug_retvals = {
                "max_aug_reward": max_aug_reward,
                "best_aug_actions": get_best_actions(out["actions"], max_idxs)
                if return_action
                else None,
            }

        return {**out, **loss_retvals, **multi_start_retvals, **aug_retvals}
