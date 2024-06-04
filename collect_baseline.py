import argparse
import numpy as np
import timeit

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.routing.cvrp.baselines.solve import solve

# Config
phase = "test"
task = "vrp"
size = "50"
solver = "pyvrp"
max_runtime = 1 # How long the solver is allowed to run for one instance
num_procs = 1 # Number of processers to use to solve instances in parallel
num_instances = 1000 # How many instances to solve

# Print the setting
print(f"Task: {task}, Size: {size}, Max runtime: {max_runtime}, Solver: {solver}")
print(f"Expected Running Time: {max_runtime * num_instances / num_procs:.1f} seconds")

start_time = timeit.default_timer()

# Load instances
td = load_npz_to_tensordict(f"data/{task}/{task}{size}_{phase}_seed1234.npz") # TODO: hardcode the seed for now

# Slice instances
td = td[:min(num_instances, td.batch_size[0])]

# Solve instances
actions, costs = solve(
    instances=td,
    max_runtime=max_runtime,
    num_procs=num_procs,
    # data_type="mtvrp",
    data_type="cvrp",
    solver=solver,
)

end_time = timeit.default_timer()

print(f"Real runtime: {end_time - start_time}")
print(f"Average cost: {costs.mean()}")

# Save the actions and costs to npz files
np.savez(f"data/{task}/{phase}/{size}_sol_{solver}.npz", actions=actions.numpy(), costs=costs.numpy())

# If not exist, create a file to log the real running time
with open(f"data/{task}/{phase}/{size}_sol_{solver}.txt", "a") as f:
    f.write("real_time\n")
