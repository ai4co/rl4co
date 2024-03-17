from typing import Optional, Union

import torch.nn as nn

from tensordict import TensorDict

try:
    from torch_geometric.data import Batch
except ImportError:
    # `Batch` is referred to only as type notations in this file
    Batch = None

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.zoo.common.nonautoregressive.decoder import NonAutoregressiveDecoder
from rl4co.models.zoo.deepaco.antsystem import AntSystem


class DeepACODecoder(NonAutoregressiveDecoder):
    """TODO"""

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int,
        num_layers: int,
        heatmap_generator: Optional[nn.Module] = None,
        linear_bias: bool = True,
        aco_class=AntSystem,
        **aco_args,
    ) -> None:
        super(DeepACODecoder, self).__init__(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            heatmap_generator=heatmap_generator,
            linear_bias=linear_bias,
        )
        self.aco_class = aco_class
        self.aco_args = aco_args

    def forward(
        self,
        td_initial: TensorDict,
        graph: Batch,  # type: ignore
        env: Union[str, RL4COEnvBase, None] = None,
        calc_reward: bool = True,
        n_ants: Optional[int] = None,
        n_iterations: Optional[int] = None,
        phase: str = "train",
        **unused_kwargs,
    ):
        """TODO"""
        if phase == "train":
            # use the procedure inherited from NonAutoregressiveDecoder for training
            return super().forward(
                td_initial,
                graph,
                env,
                decode_type="multistart_sampling",
                calc_reward=calc_reward,
                phase=phase,
            )

        # Instantiate environment if needed
        if env is None or isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # calculate heatmap
        heuristic_logp = self.heatmap_generator(graph)

        aco_args = self.aco_args.copy()
        if n_ants is not None:
            aco_args["n_ants"] = n_ants

        aco = self.aco_class(heuristic_logp, **aco_args)
        td, actions, reward = aco.run(td_initial, env, n_iterations)
        td.set("reward", reward)

        return None, actions, td
