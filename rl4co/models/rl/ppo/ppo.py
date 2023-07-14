from typing import Any, Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PPO(RL4COLitModule):
    """
    PPO -> TODO
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: nn.Module,
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 2,  # K
        mini_batch_size: Union[int, float] = 0.25,  # 0.25,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = 0.5,  # max gradient norm
        **kwargs,
    ):
        kwargs["automatic_optimization"] = False  # PPO uses custom optimization routine
        super().__init__(env, policy, **kwargs)
        self.critic = critic

    def configure_optimizers(self):
        pass
