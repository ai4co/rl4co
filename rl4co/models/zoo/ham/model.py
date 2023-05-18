from rl4co.models.rl.reinforce.baselines import WarmupBaseline, RolloutBaseline
from rl4co.models.zoo.ham.policy import HeterogeneousAttentionModelPolicy
from rl4co.models.rl.reinforce.base import REINFORCE


class HeterogeneousAttentionModel(REINFORCE):
    """Heterogenous Attention Model for solving the Pickup and Delivery Problem based on REINFORCE
    https://arxiv.org/abs/2110.02634

    Args:
        env: TorchRL Environment
        policy: Policy
        baseline: REINFORCE Baseline
    """

    def __init__(self, env, policy=None, baseline=None):
        super(HeterogeneousAttentionModel, self).__init__(env, policy, baseline)
        assert (
            self.env.name == "pdp"
        ), "HeterogeneousAttentionModel only works for PDP (Pickup and Delivery Problem)"
        self.policy = (
            HeterogeneousAttentionModelPolicy(self.env) if policy is None else policy
        )
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )
