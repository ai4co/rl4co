from os.path import join as pjoin
from typing import Iterable, Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from rl4co.data.dataset import TensorDictDataset
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RL4COEnvBase(EnvBase):
    """Base class for RL4CO environments based on TorchRL EnvBase.
    The environment has the usual methods for stepping, resetting, and getting the specifications of the environment
    that shoud be implemented by the subclasses of this class.
    It also has methods for getting the reward, action mask, and checking the validity of the solution, and
    for generating and loading the datasets (supporting multiple dataloaders as well for validation and testing).

    Args:
        data_dir: Root directory for the dataset
        train_file: Name of the training file
        val_file: Name of the validation file
        test_file: Name of the test file
        val_dataloader_names: Names of the dataloaders to use for validation
        test_dataloader_names: Names of the dataloaders to use for testing
        check_solution: Whether to check the validity of the solution at the end of the episode
        dataset_cls: Dataset class to use for the environment (which can influence performance)
        seed: Seed for the environment
        device: Device to use. Generally, no need to set as tensors are updated on the fly
        batch_size: Batch size to use for the environment. Generally, no need to set as tensors are updated on the fly
        run_type_checks: If True, run type checks on the TensorDicts at each step
        allow_done_after_reset: If True, an environment can be done after a reset
        _torchrl_mode: Whether to use the TorchRL mode (see :meth:`step` for more details)
    """

    batch_locked = False

    def __init__(
        self,
        *,
        data_dir: str = "data/",
        train_file: str = None,
        val_file: str = None,
        test_file: str = None,
        val_dataloader_names: list = None,
        test_dataloader_names: list = None,
        check_solution: bool = True,
        dataset_cls: callable = TensorDictDataset,
        seed: int = None,
        device: str = "cpu",
        batch_size: torch.Size = None,
        run_type_checks: bool = False,
        allow_done_after_reset: bool = False,
        _torchrl_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            run_type_checks=run_type_checks,
            allow_done_after_reset=allow_done_after_reset,
        )
        # if any kwargs are left, we want to warn the user
        if kwargs:
            log.warning(
                f"Unused keyword arguments: {', '.join(kwargs.keys())}. "
                "Please check the documentation for the correct keyword arguments"
            )
        self.data_dir = data_dir
        self.train_file = pjoin(data_dir, train_file) if train_file is not None else None
        self._torchrl_mode = _torchrl_mode
        self.dataset_cls = dataset_cls

        def get_files(f):
            if f is not None:
                if isinstance(f, Iterable) and not isinstance(f, str):
                    return [pjoin(data_dir, _f) for _f in f]
                else:
                    return pjoin(data_dir, f)
            return None

        def get_multiple_dataloader_names(f, names):
            if f is not None:
                if isinstance(f, Iterable) and not isinstance(f, str):
                    if names is None:
                        names = [f"{i}" for i in range(len(f))]
                    else:
                        assert len(names) == len(
                            f
                        ), "Number of dataloader names must match number of files"
                else:
                    if names is not None:
                        log.warning(
                            "Ignoring dataloader names since only one dataloader is provided"
                        )
            return names

        self.val_file = get_files(val_file)
        self.test_file = get_files(test_file)
        self.val_dataloader_names = get_multiple_dataloader_names(
            self.val_file, val_dataloader_names
        )
        self.test_dataloader_names = get_multiple_dataloader_names(
            self.test_file, test_dataloader_names
        )
        self.check_solution = check_solution
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def step(self, td: TensorDict) -> TensorDict:
        """Step function to call at each step of the episode containing an action.
        If `_torchrl_mode` is True, we call `_torchrl_step` instead which set the
        `next` key of the TensorDict to the next state - this is the usual way to do it in TorchRL,
        but inefficient in our case
        """
        if not self._torchrl_mode:
            # Default: just return the TensorDict without farther checks etc is faster
            td = self._step(td)
            return {"next": td}
        else:
            # Since we simplify the syntax
            return self._torchrl_step(td)

    def _torchrl_step(self, td: TensorDict) -> TensorDict:
        """See :meth:`super().step` for more details.
        This is the usual way to do it in TorchRL, but inefficient in our case

        Note:
            Here we clone the TensorDict to avoid recursion error, since we allow
            for directly updating the TensorDict in the step function
        """
        # sanity check
        self._assert_tensordict_shape(td)
        next_preset = td.get("next", None)

        next_tensordict = self._step(
            td.clone()
        )  # NOTE: we clone to avoid recursion error
        next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            next_tensordict.update(next_preset.exclude(*next_tensordict.keys(True, True)))
        td.set("next", next_tensordict)
        return td

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
                if isinstance(f, Iterable) and not isinstance(f, str):
                    names = getattr(self, f"{phase}_dataloader_names")
                    return {
                        name: self.dataset_cls(self.load_data(_f, batch_size))
                        for name, _f in zip(names, f)
                    }
                else:
                    td = self.load_data(f, batch_size)
            except FileNotFoundError:
                log.error(
                    f"Provided file name {f} not found. Make sure to provide a file in the right path first or "
                    f"unset {phase}_file to generate data automatically instead"
                )
                td = self.generate_data(batch_size)

        return self.dataset_cls(td)

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

    def to(self, device):
        """Override `to` device method for safety against `None` device (may be found in `TensorDict`))"""
        if device is None:
            return self
        else:
            return super().to(device)

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
