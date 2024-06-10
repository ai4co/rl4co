import lkh
import numpy as np

from tensordict import TensorDict
from torch import Tensor

from .utils import scale

LKH_SCALING_FACTOR = 100_000


def solve(
    instance: TensorDict,
    max_runtime: float,
    problem_type: str,
    num_runs: int,
    solver_loc: str,
) -> tuple[Tensor, Tensor]:
    """
    Solves an CVRP instance with OR-Tools.

    Parameters
    ----------
    instance
        The CVRP instance to solve.
    max_runtime
        The maximum runtime for the solver.
    problem_type
        The problem type for LKH3.
    num_runs
        The number of runs to perform and returns the best result.
    solver_loc
        The location of the LKH3 solver executable.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    problem = instance2problem(instance, problem_type, LKH_SCALING_FACTOR)
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
    routes, cost = lkh.solve(
        solver_loc,
        problem=problem,
        time_limit=max_runtime,
        runs=num_runs,
    )

    action = routes2action(routes)
    return action, cost


def instance2problem(
    instance: TensorDict,
    problem_type: str,
    scaling_factor,
) -> lkh.LKHProblem:
    """
    Converts an CVRP instance to an LKHProblem instance.

    Parameters
    ----------
    instance
        The CVRP instance to convert.
    problem_type
        The problem type for LKH3. See ``constants.ROUTEFINDER2LKH`` for
        supported problem types.
    scaling_factor
        The scaling factor to apply to the instance data.
    """
    num_locations = instance["demand_linehaul"].size()[0]

    # Data specifications
    specs = {}
    specs["DIMENSION"] = num_locations
    specs["CAPACITY"] = scale(instance["vehicle_capacity"], scaling_factor)

    specs["EDGE_WEIGHT_TYPE"] = "EXPLICIT"
    specs["EDGE_WEIGHT_FORMAT"] = "FULL_MATRIX"
    specs["NODE_COORD_TYPE"] = "TWOD_COORDS"

    # LKH can only solve VRP variants that are explicitly supported (so no
    # arbitrary combinations between individual supported features). We can
    # support some open variants with some modeling tricks.
    lkh_problem_type = "CVRP"
    specs["TYPE"] = lkh_problem_type

    # Data sections
    sections = {}
    sections["NODE_COORD_SECTION"] = scale(instance["locs"], scaling_factor)

    demand = scale(instance["demand"], scaling_factor)
    sections["DEMAND_SECTION"] = demand

    distances = instance["cost_matrix"]

    sections["EDGE_WEIGHT_SECTION"] = scale(distances, scaling_factor)

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
