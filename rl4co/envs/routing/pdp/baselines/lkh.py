import os
import lkh
import numpy as np
import torch

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs.routing.tsp.baselines.lkh import NUM_RUNS
from rl4co.utils.ops import get_distance_matrix

cwd = os.getcwd()

LKH_SCALING_FACTOR = 100_000
MAX_RUNS = 100
SOLVER_LOC = os.path.abspath(os.path.join(cwd, "../LKH-3.0.9/LKH"))


def _scale(data: Tensor, scaling_factor: int):
    """
    Scales ands rounds data to integers so PyVRP can handle it.
    """
    array = (data * scaling_factor).numpy().round()
    array = np.where(array == np.inf, np.iinfo(np.int32).max, array)
    array = array.astype(int)

    if array.size == 1:
        return array.item()

    return array


def solve(
    instance: TensorDict,
    max_runtime: float,
) -> tuple[Tensor, Tensor]:
    """
    Solves an AnyVRP instance with OR-Tools.

    Parameters
    ----------
    instance
        The AnyVRP instance to solve.
    max_runtime
        The maximum runtime for the solver.
    num_runs
        The number of runs to perform and returns the best result.
    solver_loc
        The location of the LKH3 solver executable.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    problem = instance2problem(instance, LKH_SCALING_FACTOR)
    action, cost = _solve(problem, max_runtime)
    cost /= -LKH_SCALING_FACTOR

    return action, cost


def _solve(
    problem: lkh.LKHProblem,
    max_runtime: float,
) -> tuple[Tensor, Tensor]:
    """
    Solves an instance with LKH3.

    Parameters
    ----------
    problem
        The LKHProblem instance.
    max_runtime
        The maximum runtime for each solver run.
    num_runs
        The number of runs to perform and returns the best result.
        Note: Each run uses a different initial solution. LKH has difficulty
        finding feasible solutions, so performing more runs can help to find
        solutions that are feasible.
    solver_loc
        The location of the LKH3 solver executable.
    """
    route = lkh.solve(
        solver=SOLVER_LOC,
        problem=problem,
        time_limit=max_runtime,
        runs=NUM_RUNS,
    )[0]
    route = route2actions(route)
    cost = route2costs(route, problem)
    return route, cost


def instance2problem(
    instance: TensorDict,
    scaling_factor,
) -> lkh.LKHProblem:
    """
    Converts an AnyVRP instance to an LKHProblem instance.

    Parameters
    ----------
    instance
        The AnyVRP instance to convert.
    scaling_factor
        The scaling factor to apply to the instance data.
    """
    num_locations = instance["locs"].size(0)

    # available fields: depot, locs
    locs = torch.cat((instance["depot"].unsqueeze(0), instance["locs"]), dim=0)

    # Data specifications
    specs = {}
    specs["DIMENSION"] = num_locations + 1

    specs["EDGE_WEIGHT_TYPE"] = "EXPLICIT"
    specs["EDGE_WEIGHT_FORMAT"] = "FULL_MATRIX"
    specs["NODE_COORD_TYPE"] = "TWOD_COORDS"

    specs["TYPE"] = "PDTSP"

    # pickups and deliveries
    pdp_matrix = np.zeros((num_locations + 1, 6))
    # pdp_matrix[:, 0] = np.arange(num_locations + 1) + 1  # Start from 1
    pdp_matrix[1 : num_locations // 2 + 1, -2] = (
        np.arange(num_locations // 2 + 1, num_locations + 1) + 1
    )
    pdp_matrix[num_locations // 2 + 1 :, -1] = np.arange(1, num_locations // 2 + 1) + 1
    pdp_matrix = pdp_matrix.astype(int)

    # Data sections
    sections = {}
    sections["NODE_COORD_SECTION"] = _scale(locs, scaling_factor)  # includes the depot
    sections["PICKUP_AND_DELIVERY_SECTION"] = pdp_matrix
    sections["EDGE_WEIGHT_SECTION"] = _scale(get_distance_matrix(locs), scaling_factor)

    # Convert to VRPLIB-like string.
    problem = "\n".join(f"{k} : {v}" for k, v in specs.items())
    problem += "\n" + "\n".join(_format(name, data) for name, data in sections.items())
    problem += "\n" + "\n".join(["DEPOT_SECTION", "1", "-1", "EOF"])

    return lkh.LKHProblem.parse(problem)


def _is_1D(data) -> bool:
    for elt in data:
        if isinstance(elt, (list, tuple, np.ndarray)):
            return False
    return True


def _format(name: str, data) -> str:
    """
    Formats a data section.

    Parameters
    ----------
    name
        The name of the section.
    data
        The data to be formatted.

    Returns
    -------
    str
        A VRPLIB-formatted data section.
    """
    section = [name]
    include_idx = name not in ["EDGE_WEIGHT_SECTION", "BACKHAUL_SECTION"]

    if name == "BACKHAUL_SECTION":
        # Treat backhaul section as row vector.
        section.append("\t".join(str(val) for val in data))

    elif _is_1D(data):
        # Treat 1D arrays as column vectors, so each element is a row.
        for idx, elt in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            section.append(prefix + str(elt))
    else:
        for idx, row in enumerate(data, 1):
            prefix = f"{idx}\t" if include_idx else ""
            rest = "\t".join([str(elt) for elt in row])
            section.append(prefix + rest)

    return "\n".join(section)


def routes2action(routes: list[list[int]]) -> list[int]:
    """
    Converts LKH routes to an action.
    """
    # LKH routes are location-indexed, which in turn are 1-indexed. The first
    # location is always the depot, so we subtract 2 to get client indices.
    # LKH routes are 1-indexed, so we subtract 1 to get client indices.
    routes_ = [[client - 1 for client in route] for route in routes]
    return [visit for route in routes_ for visit in route + [0]]


def route2actions(route: list[int]) -> list[int]:
    return [1] + route + [1]


def route2costs(route: list[list[int]], problem: lkh.LKHProblem) -> Tensor:
    """
    Computes the costs of a route.
    """
    cost = 0
    for ii in range(len(route) - 1):
        cost += problem.edge_weights[route[ii] - 1][route[ii + 1] - 1]
    return cost
