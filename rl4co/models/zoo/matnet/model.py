from typing import Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.matnet.policy import MatNetPolicy, MultiStageMatNetPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def select_policy(env: RL4COEnvBase, **policy_params):
    if env.name == "XXXXXXXffsp":
        return MultiStageMatNetPolicy(env_name=env.name, **policy_params)
    else:
        return MatNetPolicy(env_name=env.name, **policy_params)


class MatNet(POMO):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, MatNetPolicy] = None,
        num_starts: int = None,
        policy_params: dict = {},
        **kwargs,
    ):
        if policy is None:
            policy = select_policy(env, **{"stage_cnt": env.num_stage, **policy_params})

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

        super(MatNet, self).__init__(
            env=env,
            policy=policy,
            num_starts=num_starts,
            **kwargs,
        )

    def set_decode_type_multistart(self, phase: str):
        """Overwrites POMOs set_decode_type_multistart function to account for MultiStageMatNet policy.
        Since the parent policy defines sub policies, the decode types of these sub policies have to be
        updated with a multistart_ flag as well.
        For all other models, this function only executes the parents classes set_decode_type_multistart fn.
        """
        super().set_decode_type_multistart(phase)
        if isinstance(self.policy, MultiStageMatNetPolicy):
            for sub_policy in self.policy.stage_models:
                sub_policy.__dict__.update(
                    {
                        "train_decode_type": self.policy.train_decode_type,
                        "val_decode_type": self.policy.val_decode_type,
                        "test_decode_type": self.policy.test_decode_type,
                    }
                )
