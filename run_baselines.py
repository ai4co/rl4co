import os
import timeit

from functools import partial
from multiprocessing import Pool

import numpy as np

from tensordict import TensorDict
from torch import Tensor

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.routing.cvrp.baselines.pyvrp import solve_multipr

N_INSTANCES = 1_000

baselines = {
    # "pdp": "lkh",
    # "tsp": "lkh",
    "vrp": solve_multipr,
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


def main():
    for name in baselines:
        data_filepath = f"data/{name}/"
        files = os.listdir(data_filepath)
        print("files", files)
        for filename in files:
            if filename.endswith(".npz"):
                start_time = timeit.default_timer()
                data = load_npz_to_tensordict(f"{data_filepath}{filename}")
                instances = shorten_tensordict(data, N_INSTANCES)
                print("Run baseline for", filename)
                actions, costs = solve_multipr(
                    instances=instances, max_runtime=1, num_procs=24
                )
                duration = timeit.default_timer() - start_time
                print(f"Real runtime: {duration}")
                print(f"Average cost: {costs.mean()}")
                np.savez(
                    f"data/{name}/sol_{filename}.npz",
                    actions=actions.numpy(),
                    costs=costs.numpy(),
                )
                # If not exist, create a file to log the real running time
                with open(f"data/{name}/sol_{filename}.txt", "a") as f:
                    f.write(f"real_time\n{duration}")


if __name__ == "__main__":
    main()
