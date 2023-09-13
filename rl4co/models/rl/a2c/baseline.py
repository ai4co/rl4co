import torch.nn.functional as F
import torch.nn as nn

from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co import utils


log = utils.get_pylogger(__name__)


class CriticBaseline(REINFORCEBaseline):
    """Critic baseline: use critic network as baseline for REINFORCE (Policy Gradients).
    We separate A2C from REINFORCE for clarity, although they are essentially the same algorithm with different baselines.

    Args:
        critic: Critic network to use as baseline. If None, create a new critic network based on the environment
    """

    def __init__(self, critic: nn.Module = None, **unused_kw):
        super(CriticBaseline, self).__init__()
        self.critic = critic

    def setup(self, model, env, **kwargs):
        if self.critic is None:
            log.info("Creating critic network for {}".format(env.name))
            self.critic = CriticNetwork(env.name, **kwargs)

    def eval(self, x, c, env=None):
        v = self.critic(x).squeeze(-1)
        # detach v since actor should not backprop through baseline, only for neg_loss
        return v.detach(), -F.mse_loss(v, c.detach())
