import lkh
import numpy as np
import torch

from tensordict import TensorDict
from torch import Tensor

from rl4co.utils.ops import get_distance_matrix

from .utils import _scale

LKH_SCALING_FACTOR = 100_000


def solve(
    instance: TensorDict,
    max_runtime: float,
    solver_loc: str,
    num_runs: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Solves instance with LKH3.

    Args:
        instance: The PDP instance to solve.
        max_runtime: The maximum runtime for the solver.
        solver_loc: The location of the LKH3 solver executable.
        num_runs: The number of runs to perform and returns the best result.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    problem = instance2problem(instance, LKH_SCALING_FACTOR)
    action, cost = _solve(problem, max_runtime, num_runs, solver_loc)
    cost /= -LKH_SCALING_FACTOR

    return action, cost


def _solve(
    problem: lkh.LKHProblem,
    max_runtime: float,
    num_runs: int,
    solver_loc: str,
) -> tuple[Tensor, Tensor]:
    """
    Solves an instance with LKH3.

    Args:
        problem: The LKHProblem instance.
        max_runtime: The maximum runtime for each solver run.
        num_runs: The number of runs to perform and returns the best result.
        solver_loc: The location of the LKH3 solver executable.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    routes, cost = lkh.solve(
        solver_loc,
        problem=problem,
        time_limit=max_runtime,
        runs=num_runs,
    )

    action = routes2actions(routes)
    cost = route2costs(routes, problem)
    return action, cost


def instance2problem(
    instance: TensorDict,
    scaling_factor,
) -> lkh.LKHProblem:
    """
    Converts an AnyVRP instance to an LKHProblem instance.

    Args:
        instance: The AnyVRP instance to convert.
        scaling_factor: The scaling factor to apply to the instance data.

    Returns:
        The LKHProblem instance.
    """

    # If we have action_mask, then env has been reset
    if "action_mask" in instance:
        num_locations = instance["locs"].size(0) - 1  # exclude depot
        locs = instance["locs"]
    else:
        num_locations = instance["locs"].size(0)
        # available fields: depot, locs
        locs = torch.cat((instance["depot"][None], instance["locs"]), dim=0)

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
    pdp_matrix[1 : num_locations // 2 + 1, -1] = (
        np.arange(num_locations // 2 + 1, num_locations + 1) + 1
    )
    pdp_matrix[num_locations // 2 + 1 :, -2] = np.arange(1, num_locations // 2 + 1) + 1
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

    Args:
        name: The name of the section.
        data: The data to be formatted.

    Returns:
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


def routes2actions(routes: list[list[int]]) -> list[int]:
    """
    Converts LKH routes to an action.
    """
    # LKH routes are location-indexed, which in turn are 1-indexed. The first
    # location is always the depot, so we subtract 2 to get client indices.
    # LKH routes are 1-indexed, so we subtract 1 to get client indices.
    routes_ = [[client - 1 for client in route] for route in routes]
    return [visit for route in routes_ for visit in route] + [0]


def route2costs(route: list[list[int]], problem: lkh.LKHProblem) -> Tensor:
    """
    Computes the costs of a route.
    """
    cost = 0
    for ii in range(len(route) - 1):
        cost += problem.edge_weights[route[ii] - 1][route[ii + 1] - 1]
    return cost
