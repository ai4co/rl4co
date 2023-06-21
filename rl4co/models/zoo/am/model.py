from rl4co.models.rl.reinforce.base import REINFORCE
from rl4co.models.rl.reinforce.baselines import RolloutBaseline, WarmupBaseline
from rl4co.models.zoo.am.policy import AttentionModelPolicy


class AttentionModel(REINFORCE):
    """
    Attention Model for neural combinatorial optimization based on REINFORCE
    Based on Wouter Kool et al. (2018) https://arxiv.org/abs/1803.08475
    Refactored from reference implementation: https://github.com/wouterkool/attention-learn-to-route

    Args:
        env: TorchRL Environment
        policy: Policy
        baseline: REINFORCE Baseline
    """

    def __init__(self, env, policy=None, baseline=None, **policy_kwargs):
        super(AttentionModel, self).__init__(env, policy, baseline)
        self.policy = (
            AttentionModelPolicy(self.env, **policy_kwargs) if policy is None else policy
        )
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )
