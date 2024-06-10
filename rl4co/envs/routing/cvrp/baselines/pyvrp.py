import torch
import numpy as np
import pyvrp as pyvrp

from torch import Tensor
from tensordict.tensordict import TensorDict
from pyvrp.stop import MaxRuntime
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve

from rl4co.utils.ops import get_distance_matrix

from .utils import scale

PYVRP_SCALING_FACTOR = 1_000


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves the CVRP instance with PyVRP.

    Parameters
    ----------
    instance
        The CVRP instance to solve.
    max_runtime
        Maximum runtime for the solver.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    data = instance2data(instance, PYVRP_SCALING_FACTOR)
    stop = MaxRuntime(max_runtime)
    result = _solve(data, stop)

    solution = result.best
    action = solution2action(solution)
    cost = -result.cost() / PYVRP_SCALING_FACTOR

    return action, cost


def instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
    """
    Converts an CVRP instance to a ProblemData instance.

    Parameters
    ----------
    instance
        The CVRP instance to convert.

    Returns
    -------
    ProblemData
        The ProblemData instance.
    """
    num_locs = instance["demand"].size()[0]

    locs = instance["locs"]

    coords = scale(locs, scaling_factor)
    matrix = scale(get_distance_matrix(locs), scaling_factor)

    capacity = scale(instance["vehicle_capacity"], scaling_factor)
    demand = scale(instance["demand"], scaling_factor)
    depot = Depot(
        x=coords[0][0],
        y=coords[0][1],
    )
    clients = [
        Client(
            x=coords[idx + 1][0],
            y=coords[idx + 1][1],
            delivery=demand[idx],
        )
        for idx in range(num_locs)
    ]

    vehicle_type = VehicleType(
        num_available=num_locs - 1,  # one vehicle per client
        capacity=capacity,
    )

    return ProblemData(clients, [depot], [vehicle_type], matrix, matrix)


def solution2action(solution: pyvrp.Solution) -> list[int]:
    """
    Converts a PyVRP solution to the action representation, i.e., a giant tour.
    """
    return [visit for route in solution.routes() for visit in route.visits() + [0]]
