import numpy as np
import pyvrp as pyvrp
import torch

from loguru import logger
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
from pyvrp.stop import MaxRuntime
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import get_distance_matrix

PYVRP_SCALING_FACTOR = 1_000


def solve_instance(
    instance: TensorDict, max_runtime: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solves the AnyVRP instance with PyVRP.

    Parameters
    ----------
    instance
        The AnyVRP instance to solve.
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


def _scale(data: torch.Tensor, scaling_factor: int):
    """
    Scales ands rounds data to integers so PyVRP can handle it.
    """
    array = (data * scaling_factor).numpy().round()
    array = np.where(array == np.inf, np.iinfo(np.int32).max, array)
    array = array.astype(int)

    if array.size == 1:
        return array.item()

    return array


def instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
    """
    Converts an AnyVRP instance to a ProblemData instance.

    Parameters
    ----------
    instance
        The AnyVRP instance to convert.

    Returns
    -------
    ProblemData
        The ProblemData instance.
    """
    num_locs = instance["demand"].size()[0]

    locs = torch.cat((instance["depot"].unsqueeze(0), instance["locs"]), dim=0)

    coords = _scale(locs, scaling_factor)
    matrix = _scale(get_distance_matrix(locs), scaling_factor)

    capacity = _scale(instance["capacity"], scaling_factor)
    demand = _scale(instance["demand"], scaling_factor)
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
