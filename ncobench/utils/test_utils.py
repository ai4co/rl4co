import torch
from torch.utils.data import DataLoader

from ncobench.envs.tsp import TSPEnv
from ncobench.data.dataset import TorchDictDataset


def get_env(env_name, size):
    if env_name == "tsp":
        env = TSPEnv(num_loc=size)
    else:
        raise NotImplementedError

    return env.transform()


def generate_env_data(env, size):
    env = get_env(env, size)
    init_td = env.reset(batch_size=[2])
    dataset = TorchDictDataset(init_td)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False, 
        num_workers=0,
        collate_fn=torch.stack, 
    )

    return env, next(iter(dataloader))