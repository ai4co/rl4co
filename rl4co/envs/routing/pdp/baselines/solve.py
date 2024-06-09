from functools import partial
from multiprocessing import Pool

from tensordict.tensordict import TensorDict
from torch import Tensor


class NoSolver:
    def solve(self, *args, **kwargs):
        pass


try:
    from .lkh3 import solve as lkh
except ImportError:
    lkh = NoSolver()


def solve(
    instances: TensorDict,
    max_runtime: float,
    num_procs: int = 1,
    solver: str = "lkh",
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instances with PyVRP.

    Args:
        instances: The AnyVRP instances to solve.
        max_runtime: The maximum runtime for the solver.
        num_procs: The number of processes to use.
        solver: The solver to use.

    Returns:
        A tuple containing the action and the cost, respectively.
    """

    if solver == "lkh" and isinstance(lkh, NoSolver):
        raise ImportError(
            "LKH is not installed. Please install it using `GIT_LFS_SKIP_SMUDGE=1 pip install git+https://github.com/leonlan/pylkh.git@7ba9965`"
        )

    solvers = {"lkh": lkh}
    if solver not in solvers:
        raise ValueError(f"Unknown baseline solver: {solver}")

    _solve = solvers[solver]
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
