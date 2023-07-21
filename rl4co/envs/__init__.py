# Base environment
# Main Environments
from rl4co.envs.atsp import ATSPEnv
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.cvrp import CVRPEnv
from rl4co.envs.dpp import DPPEnv
from rl4co.envs.ffsp import FFSPEnv
from rl4co.envs.mdpp import MDPPEnv
from rl4co.envs.mtsp import MTSPEnv
from rl4co.envs.op import OPEnv
from rl4co.envs.pctsp import PCTSPEnv
from rl4co.envs.pdp import PDPEnv
from rl4co.envs.sdvrp import SDVRPEnv
from rl4co.envs.spctsp import SPCTSPEnv
from rl4co.envs.tsp import TSPEnv

# Register environments
ENV_REGISTRY = {
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "dpp": DPPEnv,
    "mdpp": MDPPEnv,
    "mtsp": MTSPEnv,
    "op": OPEnv,
    "pctsp": PCTSPEnv,
    "pdp": PDPEnv,
    "sdvrp": SDVRPEnv,
    "spctsp": SPCTSPEnv,
    "tsp": TSPEnv,
}


def get_env(env_name: str, *args, **kwargs) -> RL4COEnvBase:
    """Get environment by name.

    Args:
        env_name: Environment name
        *args: Positional arguments for environment
        **kwargs: Keyword arguments for environment

    Returns:
        Environment
    """
    env_cls = ENV_REGISTRY.get(env_name, None)
    if env_cls is None:
        raise ValueError(
            f"Unknown environment {env_name}. Available environments: {ENV_REGISTRY.keys()}"
        )
    return env_cls(*args, **kwargs)
