from typing import Optional, Callable, Union

import torch

from torch.distributions import Uniform, Normal, Exponential, Poisson
from tensordict.tensordict import TensorDictBase, TensorDict
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


class Generator():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size)

    def _generate(self, batch_size, **kwargs) -> TensorDict:
        raise NotImplementedError


def get_sampler(
        val_name: str,
        distribution: Union[int, float, str, type, Callable],
        min_val: float,
        max_val: float,
        **kwargs
    ):
    """Get the sampler for the variable with the given distribution
    Args:
        val_name: Name of the variable
        distribution: int/float value (as constant distribution), or string with the distribution name (supporting 
            uniform, normal, exponential, and poisson) or PyTorch Distribution type or a callable function that
            returns a PyTorch Distribution
        min_val: Minimum value for the variable, used for Uniform distribution
        max_val: Maximum value for the variable, used for Uniform distribution
        kwargs: Additional arguments for the distribution
    """
    if isinstance(distribution, (int, float)):
        return Uniform(low=distribution, high=distribution)
    elif distribution == "center": # Depot
        return Uniform(low=(max_val-min_val)/2, high=(max_val-min_val)/2)
    elif distribution == "corner": # Depot
        return Uniform(low=min_val, high=min_val)
    elif distribution == Uniform or distribution == "uniform":
        return Uniform(low=min_val, high=max_val)
    elif distribution == Normal or distribution == "normal":
        assert kwargs.get("mean_"+val_name, None) is not None, "mean is required for Normal distribution"
        assert kwargs.get(val_name+"_std", None) is not None, "std is required for Normal distribution"
        return Normal(mean=kwargs[val_name+"_mean"], std=kwargs[val_name+"_std"])
    elif distribution == Exponential or distribution == "exponential":
        assert kwargs.get(val_name+"_rate", None) is not None, "rate is required for Exponential/Poisson distribution"
        return Exponential(rate=kwargs[val_name+"_rate"])
    elif distribution == Poisson or distribution == "poisson":
        assert kwargs.get(val_name+"_rate", None) is not None, "rate is required for Exponential/Poisson distribution"
        return Poisson(rate=kwargs[val_name+"_rate"])
    elif isinstance(distribution, Callable):
        return distribution(**kwargs)
    else:
        raise ValueError(f"Invalid distribution type of {distribution}")
