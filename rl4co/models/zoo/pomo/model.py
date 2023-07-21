from tensordict import TensorDict

from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.rl.reinforce.baselines import SharedBaseline
from rl4co.models.zoo.pomo.augmentations import StateAugmentation
from rl4co.models.zoo.pomo.policy import POMOPolicy
from rl4co.utils.ops import gather_by_index, unbatchify


class POMO(REINFORCE):
    """POMO Model for neural combinatorial optimization based on REINFORCE
    Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011

    Args:
        env: TorchRL Environment
        policy: Policy
        baseline: REINFORCE Baseline
        num_starts: Number of starting actions (POMO samples) (default: 10)
        num_augment: Number of augmentations (default: 8)
    """

    def __init__(
        self,
        env,
        policy=None,
        baseline=None,
        num_starts=10,
        num_augment=8,
        **policy_kwargs,
    ):
        super(POMO, self).__init__(env, policy, baseline)
        self.policy = (
            POMOPolicy(self.env, num_starts=num_starts, **policy_kwargs)
            if policy is None
            else policy
        )

        # TODO: check baseline
        self.baseline = SharedBaseline() if baseline is None else baseline

        # POMO parameters
        self.num_augment = num_augment
        self.augment = (
            StateAugmentation(self.env.name, num_augment) if num_augment > 1 else None
        )

    def forward(self, td: TensorDict, phase: str = "train", extra=None, **policy_kwargs):
        """Evaluate model, get costs and log probabilities and compare with baseline"""

        # Get num_starts from policy. If single_traj, set num_starts and num_augment to 0
        num_starts = getattr(self.policy.decoder, "num_starts", 0)
        num_augment = self.num_augment
        if policy_kwargs.get("single_traj", False):
            num_starts, num_augment = 0, 0

        # during training, we do not augment the data
        if phase == "train":
            num_augment = 0
        elif num_augment > 1:
            td = self.augment(td)

        # Evaluate model, get costs and log probabilities
        out = self.policy(td, phase, **policy_kwargs)

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (num_starts, num_augment))

        # Training phase
        if phase == "train":
            assert num_starts > 1, "num_starts must be > 1 during training"

            ll = unbatchify(out["log_likelihood"], num_starts)

            # REINFORCE loss: we consider the rewards instead of costs to be consistent with the literature
            bl_val, bl_neg_loss = (
                self.baseline.eval(td, reward)  # unbatched reward
                if extra is None
                else (extra, 0)
            )

            advantage = reward - bl_val  # advantage = reward - baseline
            reinforce_loss = -(advantage * ll).mean()
            loss = reinforce_loss - bl_neg_loss
            out.update(
                {
                    "loss": loss,
                    "reinforce_loss": reinforce_loss,
                    "bl_loss": -bl_neg_loss,
                    "bl_val": bl_val,
                }
            )

        # Get best actions and rewards
        # Get POMO rewards and best actions
        if num_starts > 1:
            # Max POMO reward. Decouple augmentation and POMO
            max_reward, max_idxs = reward.max(dim=1)
            out.update({"max_reward": max_reward})

            # Reshape batch to [batch, num_starts, num_augment]
            if out.get("actions", None) is not None:
                actions = unbatchify(out["actions"], (num_starts, num_augment))
                out.update(
                    {"best_multistart_actions": gather_by_index(actions, max_idxs)}
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if num_augment > 1:
            if num_starts > 1:
                # If POMO is enabled, we use the best POMO rewards
                reward_ = max_reward
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

        return out
