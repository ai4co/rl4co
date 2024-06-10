import torch
import numpy as np

from torch import Tensor
from tensordict.tensordict import TensorDict


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves the OP instance with Compass.

    Parameters
    ----------
    instance
        The OP instance to solve.
    max_runtime
        Maximum runtime for the solver.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    raise NotImplementedError("Compass solver is not implemented yet.")

    # TODO
    action = None
    cost = None

    return action, cost
