from collections import defaultdict
from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from rl4co.data.dataset import TensorDictDataset


class RL4COEnv(EnvBase):
    batch_locked = False

    def __init__(
        self,
        *,
        seed: int = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, batch_size=[])
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        Gives the next observation, reward, done
        """
        raise NotImplementedError

    def _reset(
        self, td: Optional[TensorDict] = None, init_obs=None, batch_size=None
    ) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        raise NotImplementedError

    def _make_spec(self, td_params: TensorDict = None):
        """Make the specifications of the environment (observation, action, reward, done)"""
        raise NotImplementedError
    
    def get_reward(self, td, actions) -> TensorDict:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        raise NotImplementedError
    
    def dataset(self, batch_size):
        """Return a dataset of observations"""
        observation = self.generate_data(batch_size)
        return TensorDictDataset(observation)

    def generate_data(self, batch_size):
        """Dataset generation or loading"""
        raise NotImplementedError

    def transform(self):
        """Used for converting TensorDict variables (such as with torch.cat) efficiently
        https://pytorch.org/rl/reference/generated/torchrl.envs.transforms.Transform.html
        """
        return self

    def render(self, td):
        """Render the environment"""
        raise NotImplementedError

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator as it is not allowed by deepcopy
        """
        state = self.__dict__.copy()
        del state["rng"]
        return state

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng