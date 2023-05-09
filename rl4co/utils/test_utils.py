import torch
from torch.utils.data import DataLoader

from rl4co.data.dataset import TensorDictCollate
from rl4co.envs import TSPEnv, CVRPEnv, SDVRPEnv, DPPEnv


def get_env(env_name, size):
    if env_name == "tsp":
        env = TSPEnv(num_loc=size)
    elif env_name == "cvrp":
        env = CVRPEnv(num_loc=size)
    elif env_name == "sdvrp":
        env = SDVRPEnv(num_loc=size)
    elif env_name == "dpp":
        env = DPPEnv()
    else:
        raise ValueError(f"Unknown env_name: {env_name}")

    return env.transform()


def generate_env_data(env, size, batch_size):
    env = get_env(env, size)
    dataset = env.dataset([batch_size])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=TensorDictCollate(),
    )

    return env, next(iter(dataloader))
