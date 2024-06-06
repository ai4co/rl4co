import multiprocessing
import os
import timeit

from datetime import datetime as dt
from functools import partial

import numpy as np

from loguru import logger
from tensordict import TensorDict
from torch import Tensor

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.routing.cvrp.baselines.pyvrp import solve_instance as solve_pyvrp

# from rl4co.envs.routing.pdp.baselines.lkh import solve as solve_lkh_pdp
from rl4co.envs.routing.tsp.baselines.lkh import solve as solve_lkh_tsp

N_INSTANCES = 1_000

baselines = {
    # "pdp": solve_lkh_pdp,
    "tsp": solve_lkh_tsp,
    "vrp": solve_pyvrp,
}


def shorten_tensordict(td, new_size):
    new_dict = {}
    for key in td.keys():
        new_dict[key] = td[key][:new_size, ...]
    return TensorDict(new_dict, batch_size=[new_size])


def select_instance_from_batch(td, idx):
    new_dict = {}
    for key in td.keys():
        new_dict[key] = td[key][idx, ...]
    return TensorDict(new_dict)


def solve_multipr(
    solver: callable,
    instances: TensorDict,
    max_runtime: float,
    num_procs: int = 1,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    func = partial(solver, max_runtime=max_runtime, **kwargs)
    inst = instances.clone()
    if num_procs > 1:
        with multiprocessing.Pool(processes=num_procs) as pool:
            results = pool.map(func, inst)
    else:
        results = [func(instance) for instance in inst]

    actions, costs = zip(*results)

    # Pad to ensure all actions have the same length.
    max_len = max(len(action) for action in actions)
    actions = [action + [0] * (max_len - len(action)) for action in actions]

    return Tensor(actions).long(), Tensor(costs)


def solve_baseline(
    baseline: str,
    filepath: str,
    filename: str,
    save_to_path: str = None,
    num_procs: int = 24,
    max_runtime: float = 1,
):
    data = load_npz_to_tensordict(f"{filepath}{filename}")
    instances = shorten_tensordict(data, N_INSTANCES)
    logger.info(f"Run baseline for {filename}")

    # cut off the file extension
    filename = os.path.splitext(filename)[0]

    start_time = timeit.default_timer()
    actions, costs = solve_multipr(
        solver=baselines[baseline],
        instances=instances,
        max_runtime=max_runtime,
        num_procs=num_procs,
    )
    duration = timeit.default_timer() - start_time
    logger.info(f"Real runtime: {duration}")

    logger.info(f"Average cost: {costs.mean()}")
    save_to_path = save_to_path if save_to_path else filepath
    np.savez(
        f"{save_to_path}sol_{filename}_{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz",
        actions=actions.numpy(),
        costs=costs.numpy(),
    )


def main():
    logger.add("logs/run_baselines_{time}.log")
    logger.info(f"Shorten all data to {N_INSTANCES} instances.")

    max_runtime = 120
    num_procs = 32
    logger.info(
        f"Start running baselines with {num_procs} processes and max_runtime={max_runtime}"
    )

    for name in baselines:
        logger.info(f"Running {name}")
        data_filepath = f"data/{name}/"
        sol_dir = f"{data_filepath}/sol/"
        if not os.path.exists(sol_dir):
            os.makedirs(sol_dir)
        files = os.listdir(data_filepath)
        print("files", files)
        for filename in files:
            if filename.endswith(".npz") and not filename.startswith("sol_"):
                solve_baseline(
                    baseline=name,
                    filepath=data_filepath,
                    filename=filename,
                    save_to_path=sol_dir,
                    max_runtime=max_runtime,
                    num_procs=num_procs,
                )
    logger.info("Done.")


if __name__ == "__main__":
    main()
