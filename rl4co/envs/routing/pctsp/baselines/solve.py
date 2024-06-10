from functools import partial
from multiprocessing import Pool

from tensordict.tensordict import TensorDict
from torch import Tensor


def solve(
    instances: TensorDict,
    max_runtime: float,
    num_procs: int = 1,
    solver: str = "compass",
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """
    Solves the PCTSP instances with solvers.

    Args:
        instances: The PCTSP instances to solve.
        max_runtime: The maximum runtime for the solver.
        num_procs: The number of processes to use.
        solver: The solver to use, currently support 'ortools' solver.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    if solver == "ortools":
        from . import ortools
        _solve = ortools.solve
    else:
        raise ValueError(f"Unknown baseline solver: {solver}")

    func = partial(_solve, max_runtime=max_runtime, **kwargs)

    if num_procs > 1:
        with Pool(processes=num_procs) as pool:
            results = pool.map(func, instances)
    else:
        results = [func(instance) for instance in instances]

    actions, costs = zip(*results)

    # Pad to ensure all actions have the same length.
    max_len = max(len(action) for action in actions)
    actions = [action + [0] * (max_len - len(action)) for action in actions]

    return Tensor(actions).long(), Tensor(costs)
