from rl4co.models.zoo.ptrnet.policy import PointerNetworkPolicy
from rl4co.models.rl.reinforce.baselines import WarmupBaseline, RolloutBaseline
from rl4co.models.rl.reinforce.base import REINFORCE


class PointerNetwork(REINFORCE):
    def __init__(self, env, policy=None, baseline=None):
        """
        Pointer Network for neural combinatorial optimization based on REINFORCE
        Based on Vinyals et al. (2015) https://arxiv.org/abs/1506.03134
        Refactored from reference implementation: https://github.com/wouterkool/attention-learn-to-route

        Args:
            env: TorchRL Environment
            policy: Policy
            baseline: REINFORCE Baseline
        """
        super(PointerNetwork, self).__init__(env, policy, baseline)
        self.policy = PointerNetworkPolicy(self.env) if policy is None else policy
        self.baseline = (
            WarmupBaseline(RolloutBaseline()) if baseline is None else baseline
        )