from tensordict import TensorDict

from rl4co.models.rl.reinforce.base import REINFORCE
from rl4co.models.rl.reinforce.baselines import NoBaseline
from rl4co.models.zoo.symnco.augmentations import StateAugmentation
from rl4co.models.zoo.symnco.losses import (
    invariance_loss,
    problem_symmetricity_loss,
    solution_symmetricity_loss,
)
from rl4co.models.zoo.symnco.policy import SymNCOPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SymNCO(REINFORCE):
    """SymNCO Model for neural combinatorial optimization based on REINFORCE
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

    def __init__(
        self,
        env,
        policy=None,
        baseline=None,
        num_starts=10,
        num_augment=4,
        alpha=0.2,
        beta=1,
        augment_test=True,
        **policy_kwargs,
    ):
        super(SymNCO, self).__init__(env, policy, baseline)

        self.policy = (
            SymNCOPolicy(self.env, num_starts=num_starts, **policy_kwargs)
            if policy is None
            else policy
        )
        if baseline is not None:
            log.warn(
                "SymNCO uses shared baselines in the loss functions. Baseline argument will be ignored"
            )
        self.baseline = NoBaseline()  # baseline is calculated in the loss function

        # Multi-start parameters from policy, default to 1
        self.num_augment = num_augment
        self.augment = StateAugmentation(self.env.name)
        self.augment_test = augment_test
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta  # weight for solution symmetricity loss

    def forward(self, td: TensorDict, phase: str = "train", extra=None, **policy_kwargs):
        """Evaluate model, get costs and log probabilities and compare with baseline"""

        # Get num_starts from policy. If single_traj, set num_starts and num_augment to 0
        num_starts = getattr(self.policy.decoder, "num_starts", 0)
        num_augment = self.num_augment

        if policy_kwargs.get("single_traj", False):
            num_starts, num_augment = 0, 0

        if num_augment > 1:
            td = self.augment(td, num_augment)

        # Evaluate model, get costs and log probabilities
        out = self.policy(td, phase, **policy_kwargs)

        # Unbatchify reward to [batch_size, num_starts, num_augment].
        reward = unbatchify(out["reward"], (num_starts, num_augment))

        # Get multi-start (=POMO) rewards and best actions
        if num_starts > 1:
            # max multi-start reward
            max_reward, max_idxs = reward.max(dim=1)
            out.update({"max_reward": max_reward})

            # Reshape batch to [batch, num_starts, num_augment]
            if out.get("actions", None) is not None:
                # TODO: actions are not unbatchified correctly
                actions = unbatchify(out["actions"], (num_starts, num_augment))
                out.update(
                    {"best_multistart_actions": gather_by_index(actions, max_idxs)}
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if num_augment > 1:
            # If multistart is enabled, we use the best multistart rewards
            reward_ = max_reward if num_starts > 1 else reward
            # [batch, num_augment]
            max_aug_reward, max_idxs = reward_.max(dim=1)
            out.update({"max_aug_reward": max_aug_reward})
            if out.get("best_multistart_actions", None) is not None:
                out.update(
                    {
                        "best_aug_actions": gather_by_index(
                            out["best_multistart_actions"], max_idxs
                        )
                    }
                )

        # Get best actions and rewards
        # Main training loss
        if phase == "train":
            # [batch_size, num_starts, num_augment]
            ll = unbatchify(out["log_likelihood"], (num_starts, num_augment))

            # Calculate losses: problem symmetricity, solution symmetricity, invariance

            loss_ps = problem_symmetricity_loss(reward, ll) if num_starts > 1 else 0
            loss_ss = solution_symmetricity_loss(reward, ll) if num_augment > 1 else 0
            loss_inv = (
                invariance_loss(out["proj_embeddings"], num_augment)
                if num_augment > 1
                else 0
            )
            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv
            out.update(
                {
                    "loss": loss,
                    "loss_ss": loss_ss,
                    "loss_ps": loss_ps,
                    "loss_inv": loss_inv,
                }
            )

        return out
