from typing import Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.matnet.policy import MatNetPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MatNet(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, MatNetPolicy] = None,
        policy_params: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = MatNetPolicy(env=env, **policy_params)

        super(MatNet, self).__init__(
            env=env,
            policy=policy,
            **kwargs,
        )


class MatNetPOMO(POMO):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, MatNetPolicy] = None,
        num_starts: int = None,
        policy_params: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = MatNetPolicy(env=env, **policy_params)

        # Check if num_augment is not 0 or if diheral_8 is True
        if kwargs.get("num_augment", 0) != 0:
            log.error(
                "MatNet does not use symmetric augmentation. Setting num_augment to 0."
            )
        kwargs["num_augment"] = 0
        if kwargs.get("use_dihedral_8", True):
            log.error(
                "MatNet does not use symmetric Dihedral Augmentation. Setting use_dihedral_8 to False."
            )
        kwargs["use_dihedral_8"] = False

        super(MatNetPOMO, self).__init__(
            env=env,
            policy=policy,
            num_starts=num_starts,
            **kwargs,
        )
