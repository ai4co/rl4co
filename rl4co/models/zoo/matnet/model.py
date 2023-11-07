from typing import Any, Union
from rl4co.models.zoo.matnet.policy import MatNetPolicy

import torch.nn as nn

from rl4co.models.zoo.pomo.model import POMO
from rl4co.envs.common.base import RL4COEnvBase


class MatNet(POMO):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[nn.Module, MatNetPolicy] = None,
        optimizer_kwargs: dict = {"lr": 4 * 1e-4, "weight_decay": 1e-6},
        lr_scheduler: str = "MultiStepLR",
        lr_scheduler_kwargs: dict = {"milestones": [2001, 2101], "gamma": 0.1},
        use_dihedral_8: bool = False,
        num_starts: int = None,
        train_data_size: int = 10_000,
        batch_size: int = 200,
        policy_params: dict = {},
        model_params: dict = {},
    ):
        if policy is None:
            policy = MatNetPolicy(env_name=env.name, **policy_params)

        super(MatNet, self).__init__(
            env=env,
            policy=policy,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            use_dihedral_8=use_dihedral_8,
            num_starts=num_starts,
            train_data_size=train_data_size,
            batch_size=batch_size,
            **model_params,
        )
