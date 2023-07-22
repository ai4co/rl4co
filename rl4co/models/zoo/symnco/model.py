from typing import Any

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
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
    """SymNCO Model for neural combinatorial optimization based on REINFORCE with shared baselines
    based on Kim et al. (2022) https://arxiv.org/abs/2205.13209


    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        num_starts: Number of starts
        num_augment: Number of augmentations
        alpha: weight for invariance loss
        beta: weight for solution symmetricity loss
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: SymNCOPolicy = None,
        policy_kwargs={},
        num_augment: int = 4,
        num_starts: int = 1,
        alpha: float = 0.2,
        beta: float = 1,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy = SymNCOPolicy(env.name, **policy_kwargs)

        # Pass no baseline to superclass since there are multiple custom baselines
        super().__init__(env, policy, "no", **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        self.augment = StateAugmentation(self.env.name, num_augment=self.num_augment)
        self.alpha = alpha  # weight for invariance loss
        self.beta = beta  # weight for solution symmetricity loss

        # Add `_multistart` to decode type for train, val and test in policy if num_starts > 1
        if self.num_starts > 1:
            for phase in ["train", "val", "test"]:
                attribute = f"{phase}_decode_type"
                attr_get = getattr(self.policy, attribute)
                # If does not exist, log error
                if attr_get is None:
                    log.error(
                        f"Decode type for {phase} is None. Cannot add `_multistart`."
                    )
                    continue
                elif "multistart" in attr_get:
                    continue
                else:
                    setattr(self.policy, attribute, f"{attr_get}_multistart")

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        n_aug, n_start = self.num_augment, self.num_starts
        td = self.env.reset(batch)
        out = self.policy(td, self.env, phase=phase, num_starts=n_start)

        # Run augmentation
        if n_aug > 1:
            td = self.augment(td)

        # Unbatchify reward to [batch_size, n_start, n_aug].
        reward = unbatchify(out["reward"], (n_start, n_aug))

        # Get multi-start (=POMO) rewards and best actions
        if n_start > 1:
            # max multi-start reward
            max_reward, max_idxs = reward.max(dim=1)
            out.update({"max_reward": max_reward})

            # Reshape batch to [batch, n_start, n_aug]
            if out.get("actions", None) is not None:
                # TODO: actions are not unbatchified correctly
                actions = unbatchify(out["actions"], (n_start, n_aug))
                out.update(
                    {"best_multistart_actions": gather_by_index(actions, max_idxs)}
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if n_aug > 1:
            # If multistart is enabled, we use the best multistart rewards
            reward_ = max_reward if n_start > 1 else reward
            # [batch, n_aug]
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

        # Main training loss
        if phase == "train":
            # [batch_size, n_start, n_aug]
            ll = unbatchify(out["log_likelihood"], (n_start, n_aug))

            # Calculate losses: problem symmetricity, solution symmetricity, invariance
            loss_ps = problem_symmetricity_loss(reward, ll) if n_start > 1 else 0
            loss_ss = solution_symmetricity_loss(reward, ll) if n_aug > 1 else 0
            loss_inv = invariance_loss(out["proj_embeddings"], n_aug) if n_aug > 1 else 0
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
