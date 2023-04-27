import torch
from torch.utils.data import DataLoader

from rl4co.data.dataset import TensorDictCollate
from rl4co.envs.tsp import TSPEnv


def get_env(env_name, size):
    if env_name == "tsp":
        env = TSPEnv(num_loc=size)
    else:
        raise NotImplementedError

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
