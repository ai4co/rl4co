from typing import Any, Optional, Union

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.glop.policy import GLOPPolicy
from rl4co.utils.ops import gather_by_index, unbatchify


class GLOP(REINFORCE):
    """Implements GLOP: https://arxiv.org/abs/2312.08224

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to mean
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[GLOPPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "mean",
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = GLOPPolicy(env_name=env.name, **policy_kwargs)

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: Optional[int] = None
    ):
        td = self.env.reset(batch)
        n_samples = self.policy.n_samples

        # Evaluate policy
        out = self.policy(
            td=td,
            env=self.env,
            phase=phase,
            return_actions=True,
        )
        reward = out["reward"] = unbatchify(out["reward"], n_samples)
        max_reward, max_idxs = reward.max(dim=-1)
        out.update({"max_reward": max_reward})

        if phase == "train":
            assert n_samples > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], n_samples)
            advantage = reward - reward.mean(-1, keepdim=True)
            out["loss"] = -(advantage * log_likelihood).mean()
        else:
            if n_samples > 1 and out.get("actions", None) is not None:
                actions = unbatchify(out["actions"], n_samples)
                out.update(
                    {
                        "best_multistart_actions": gather_by_index(
                            actions, max_idxs, dim=max_idxs.dim()
                        )
                    }
                )
                out["actions"] = actions

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
