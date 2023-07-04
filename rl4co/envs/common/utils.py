from typing import Optional

import torch

from tensordict.tensordict import TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec


def make_composite_from_td(td):
    # custom funtion to convert a tensordict in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(dtype=tensor.dtype, shape=tensor.shape)
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


def batch_to_scalar(param):
    """Return first element if in batch. Used for batched parameters that are the same for all elements in the batch."""
    if len(param.shape) > 0:
        return param[0].item()
    if isinstance(param, torch.Tensor):
        return param.item()
    return param


def _set_seed(self, seed: Optional[int]):
    """Set the seed for the environment"""
    rng = torch.manual_seed(seed)
    self.rng = rng


def _getstate_env(self):
    """
    Return the state of the environment. By default, we want to avoid pickling
    the random number generator as it is not allowed by deepcopy
    """
    state = self.__dict__.copy()
    del state["rng"]
    return state
