from torch.utils.data import DataLoader

from rl4co.data.dataset import tensordict_collate_fn
from rl4co.envs import (
    CVRPEnv,
    DPPEnv,
    MDPPEnv,
    MTSPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    SDVRPEnv,
    SPCTSPEnv,
    TSPEnv,
)


def get_env(name, size):
    if name == "tsp":
        env = TSPEnv(num_loc=size)
    elif name == "cvrp":
        env = CVRPEnv(num_loc=size)
    elif name == "sdvrp":
        env = SDVRPEnv(num_loc=size)
    elif name == "pdp":
        env = PDPEnv(num_loc=size)
    elif name == "op":
        env = OPEnv(num_loc=size)
    elif name == "mtsp":
        env = MTSPEnv(num_loc=size)
    elif name == "pctsp":
        env = PCTSPEnv(num_loc=size)
    elif name == "spctsp":
        env = SPCTSPEnv(num_loc=size)
    elif name == "dpp":
        env = DPPEnv()
    elif name == "mdpp":
        env = MDPPEnv()
    else:
        raise ValueError(f"Unknown env_name: {name}")

    return env.transform()


def generate_env_data(env, size, batch_size):
    env = get_env(env, size)
    dataset = env.dataset([batch_size])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=tensordict_collate_fn,
    )

    return env, next(iter(dataloader))
