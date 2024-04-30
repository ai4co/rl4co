from typing import Optional

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.eda.dpp.env import DPPEnv
from rl4co.utils.pylogger import get_pylogger

from .generator import MDPPGenerator
from .render import render

log = get_pylogger(__name__)


class MDPPEnv(DPPEnv):
    """Multiple decap placement problem (mDPP) environment
    This is a modified version of the DPP environment where we allow multiple probing ports
    The reward can be calculated as:
        - minmax: min of the max of the decap scores
        - meansum: mean of the sum of the decap scores
    The minmax is more challenging as it requires to find the best decap location for the worst case

    Args:
        num_probes_min: minimum number of probes
        num_probes_max: maximum number of probes
        reward_type: reward type, either minmax or meansum
        td_params: TensorDict parameters
    """

    name = "mdpp"

    def __init__(
        self,
        generator: MDPPGenerator = None,
        generator_params: dict = {},
        data_dir: str = "data/dpp/",
        reward_type: str = "minmax",
        **kwargs,
    ):
        kwargs["data_dir"] = data_dir
        super().__init__(**kwargs)
        if generator is None:
            generator = MDPPGenerator(data_dir=data_dir, **generator_params)
        self.generator = generator

        assert reward_type in [
            "minmax",
            "meansum",
        ], "reward_type must be minmax or meansum"
        self.reward_type = reward_type

        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # Step function is the same as DPPEnv, only masking changes
        return super()._step(td)

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        # Other variables
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Action mask is 0 if both action_mask (e.g. keepout) and probe are 0
        action_mask = torch.logical_and(td["action_mask"], ~td["probe"])

        return TensorDict(
            {
                "locs": td["locs"],
                "probe": td["probe"],
                "i": i,
                "action_mask": action_mask,
                "keepout": ~action_mask,
            },
            batch_size=batch_size,
        )

    def get_reward(self, td, actions):
        """We call the reward function with the final sequence of actions to get the reward
        Calling per-step would be very time consuming due to decap simulation
        """
        # We do the operation in a batch
        if len(td.batch_size) == 0:
            td = td.unsqueeze(0)
            actions = actions.unsqueeze(0)

        # Reward calculation is expensive since we need to run decap simulation (not vectorizable)
        reward = torch.stack(
            [
                self._single_env_reward(td_single, action)
                for td_single, action in zip(td, actions)
            ]
        )
        return reward

    def _single_env_reward(self, td, actions):
        """Get reward for single environment. We"""

        list_probe = torch.nonzero(td["probe"]).squeeze()
        scores = torch.zeros_like(list_probe, dtype=torch.float32)
        for i, probe in enumerate(list_probe):
            # Get the decap scores for the probe location
            scores[i] = self._decap_simulator(probe, actions)
        # If minmax, return min of max decap scores else mean
        return scores.min() if self.reward_type == "minmax" else scores.mean()

    @staticmethod
    def render(td, actions=None, ax=None, legend=True, settings=None):
        return render(td, actions, ax, legend, settings)

    def _make_spec(self, generator: MDPPGenerator):
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.size**2, 2),
                dtype=torch.float32,
            ),
            probe=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            keepout=UnboundedDiscreteTensorSpec(
                shape=(generator.size**2),
                dtype=torch.bool,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.size**2),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.size**2,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)
