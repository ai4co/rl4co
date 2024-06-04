# load cvrp test instances
import os

from tensordict import TensorDict

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.routing.cvrp.baselines.pyvrp import solve as solve_pyvrp
from rl4co.utils.ops import get_distance_matrix


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


N_INSTANCES = 1_0

baselines = {
    # "pdp": "lkh",
    # "tsp": "lkh",
    "vrp": "pyvrp",
}
for name in baselines:
    data_filepath = f"data/{name}/"
    files = os.listdir(data_filepath)
    print("files", files)
    for filename in files:
        if filename.endswith(".npz"):
            data = load_npz_to_tensordict(f"{data_filepath}{filename}")
            data = shorten_tensordict(data, N_INSTANCES)
            print("Run baseline for", filename)
            # solve baseline for all instances
            for idx in range(N_INSTANCES):
                print("solve instance", idx)
                instance = select_instance_from_batch(data, idx)
                # cost matrix
                instance["cost_matrix"] = get_distance_matrix(instance["locs"])
                action, cost = solve_pyvrp(instance, 1000)
