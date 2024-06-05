import vrplib
import os
import torch

from math import ceil
from tqdm.auto import tqdm
from einops import repeat
from tensordict.tensordict import TensorDict

from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding # for q, k, v projections
from rl4co.models import POMO, AttentionModelPolicy
from rl4co.envs import CVRPEnv
from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from rl4co.utils.ops import gather_by_index
from rl4co.utils.ops import unbatchify  

device = "cuda:0"

ckpt_path = "checkpoints/pomo-cvrp50.ckpt"
model = POMO.load_from_checkpoint(ckpt_path, load_baseline=False)
model = model.to(device)

# Init environment
env = CVRPEnv(generator_params={"num_loc": 100}, check_solution=False)

# Prepare VRPLib Large scale instances
problem_names = vrplib.list_names(low=500, high=1003, vrp_type='cvrp') 

instances = [] # Collect Set A, B, E, F, M datasets
for name in problem_names:
    if "A" in name:
        instances.append(name)
    elif "B" in name:
        instances.append(name)
    elif "E" in name:
        instances.append(name)
    elif "F" in name:
        instances.append(name)
    elif "M" in name and "CMT" not in name:
        instances.append(name)
    elif "P" in name:
        instances.append(name)
    elif "X" in name:
        instances.append(name)

# Modify the path you want to save 
# Note: we don't have to create this folder in advance
path_to_save = 'data/vrplib/' 

os.makedirs(path_to_save, exist_ok=True)
for instance in tqdm(instances):
    # If we have already downloaded the instance, we don't need to download it again
    if os.path.exists(path_to_save + instance + '.vrp'):
        print(f"Instance {instance} already exists, skipping")
    else:
        vrplib.download_instance(instance, path_to_save)
    if os.path.exists(path_to_save + instance + '.sol'):
        print(f"Solution {instance} already exists, skipping")
    else:
        vrplib.download_solution(instance, path_to_save)

# Utils functions
def normalize_coord(coord:torch.Tensor) -> torch.Tensor: # if we scale x and y separately, aren't we losing the relative position of the points? i.e. we mess with the distances.
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_scaled = (x - x_min) / (x_max - x_min) 
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled 

# Main evaluate function
def evaluate(
    model,
    td,
    num_augment=8,
    num_starts=None,
):
    with torch.inference_mode():
        n_start = model.env.get_num_starts(td) if num_starts is None else num_starts

        if num_augment > 1:
            td = model.augment(td[:1])

        # Evaluate policy
        out = model.policy(
            td, model.env, phase="test", num_starts=n_start, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (num_augment, n_start))

        if n_start > 1:
            # max multi-start reward
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})

            if out.get("actions", None) is not None:
                # Reshape batch to [batch_size, num_augment, num_starts, ...]
                actions = unbatchify(out["actions"], (num_augment, n_start))
                out.update(
                    {"best_multistart_actions": gather_by_index(actions, max_idxs, dim=max_idxs.dim())}
                )
                out["actions"] = actions

        # Get augmentation score only during inference
        if num_augment > 1:
            # If multistart is enabled, we use the best multistart rewards
            reward_ = max_reward if n_start > 1 else reward
            max_aug_reward, max_idxs = reward_.max(dim=1)
            out.update({"max_aug_reward": max_aug_reward})

            if out.get("actions", None) is not None:
                actions_ = (
                    out["best_multistart_actions"] if n_start > 1 else out["actions"]
                )
                # out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})
                out.update({"best_aug_actions": actions_[max_idxs]})
                
        return out
    
# Main test loop on vrplib
gaps = []
for instance in instances:
    problem = vrplib.read_instance(os.path.join(path_to_save, instance+'.vrp'))

    coords = torch.tensor(problem['node_coord']).float()
    coords_norm = normalize_coord(coords)

    demand = torch.tensor(problem['demand'][1:]).float()
    capacity = problem['capacity']
    n = coords.shape[0]

    # Prepare the tensordict
    batch_size = 2
    
    # Tricks
    num_loc = coords.shape[-2] -1 
    generator = CVRPGenerator(num_loc=num_loc)
    env.generator = generator
    
    td = env.reset(batch_size=(batch_size,)).to(device)
    td['locs'] = repeat(coords_norm, 'n d -> b n d', b=batch_size, d=2)
    td['demand'] = repeat(demand, 'n -> b n', b=batch_size) / capacity

    num_augment = 1
    out = evaluate(model, td.clone(), num_augment=num_augment)

    # Get the best based on priority
    actions = out.get("best_aug_actions", out.get("best_multistart_actions", out.get("actions", None))).squeeze(1)
        
    # Calculate the cost on the original scale
    td['locs'] = repeat(coords, 'n d -> b n d', b=batch_size, d=2)

    if num_augment > 1:
        neg_reward = env.get_reward(td.clone()[:1], actions)
    else:
        neg_reward = env.get_reward(td.clone(), actions)
    cost = ceil(-1 * neg_reward[0].item())

    # Load the optimal cost
    solution = vrplib.read_solution(os.path.join(path_to_save, instance+'.sol'))
    optimal_cost = solution['cost']

    # Calculate the gap and print
    gap = (cost - optimal_cost) / optimal_cost
    
    gaps.append(gap)
    print(f'Problem: {instance:<15} Cost: {cost:<10} Optimal Cost: {optimal_cost:<10}\t Gap: {gap:.3%}')

    torch.cuda.empty_cache()  # Clear cache after processing
    
print(20 * "-")
print(f"Average gap: {sum(gaps) / len(gaps):.3%}")
