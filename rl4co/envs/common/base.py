from os.path import join as pjoin
from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from rl4co.data.dataset import TensorDictDataset
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RL4COEnvBase(EnvBase):
    """Base class for RL4CO environments based on TorchRL EnvBase

    Args:
        data_dir: Root directory for the dataset
        train_file: Name of the training file
        val_file: Name of the validation file
        test_file: Name of the test file
        check_solution: Whether to check the validity of the solution at the end of the episode
        seed: Seed for the environment
        device: Device to use. Generally, no need to set as tensors are updated on the fly
    """

    batch_locked = False

    def __init__(
        self,
        *,
        data_dir: str = "data/",
        train_file: str = None,
        val_file: str = None,
        test_file: str = None,
        check_solution: bool = True,
        seed: int = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device=device, batch_size=[])
        self.data_dir = data_dir
        self.train_file = pjoin(data_dir, train_file) if train_file is not None else None
        self.val_file = pjoin(data_dir, val_file) if val_file is not None else None
        self.test_file = pjoin(data_dir, test_file) if test_file is not None else None
        self.check_solution = check_solution
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        Gives the next observation, reward, done
        """
        raise NotImplementedError

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
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

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """Function to compute the action mask (feasible actions) for the current state
        Action mask is 1 if the action is feasible, 0 otherwise
        """
        raise NotImplementedError

    def check_solution_validity(self, td, actions) -> TensorDict:
        """Function to check whether the solution is valid. Can be called by the agent to check the validity of the current state
        This is called with the full solution (i.e. all actions) at the end of the episode
        """
        raise NotImplementedError

    def dataset(self, batch_size=[], phase="train", filename=None):
        """Return a dataset of observations
        Generates the dataset if it does not exist, otherwise loads it from file
        """
        if filename is not None:
            log.info(f"Overriding dataset filename from {filename}")
        f = getattr(self, f"{phase}_file") if filename is None else filename
        if f is None:
            if phase != "train":
                log.warning(f"{phase}_file not set. Generating dataset instead")
            td = self.generate_data(batch_size)
        else:
            log.info(f"Loading {phase} dataset from {f}")
            if phase == "train":
                log.warning(
                    "Loading training dataset from file. This may not be desired in RL since "
                    "the dataset is fixed and the agent will not be able to explore new states"
                )
            try:
                td = self.load_data(f, batch_size)
            except FileNotFoundError:
                log.error(
                    f"Provided file name {f} not found. Make sure to provide a file in the right path first or "
                    f"unset {phase}_file to generate data automatically instead"
                )
                td = self.generate_data(batch_size)

        return TensorDictDataset(td)

    def generate_data(self, batch_size):
        """Dataset generation"""
        raise NotImplementedError

    def transform(self):
        """Used for converting TensorDict variables (such as with torch.cat) efficiently
        https://pytorch.org/rl/reference/generated/torchrl.envs.transforms.Transform.html
        By default, we do not need to transform the environment since we use specific embeddings
        """
        return self

    def render(self, *args, **kwargs):
        """Render the environment"""
        raise NotImplementedError

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file"""
        return load_npz_to_tensordict(fpath)

    def _set_seed(self, seed: Optional[int]):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng

    def __getstate__(self):
        """Return the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        state = self.__dict__.copy()
        state["rng"] = state["rng"].get_state()
        return state

    def __setstate__(self, state):
        """Set the state of the environment. By default, we want to avoid pickling
        the random number generator directly as it is not allowed by `deepcopy`
        """
        self.__dict__.update(state)
        self.rng = torch.manual_seed(0)
        self.rng.set_state(state["rng"])
