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
        n_ants: Union[int, dict, None] = None,
        n_iterations: Union[int, dict, None] = None,
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
        self.n_ants = self._conv_params(n_ants, train=20, val=20, test=20)
        self.n_iterations = self._conv_params(n_iterations, train=5, val=20, test=50)

    def forward(
        self,
        td_initial: TensorDict,
        graph: Batch,  # type: ignore
        env: Union[str, RL4COEnvBase, None] = None,
        calc_reward: bool = True,
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
                num_starts=self.n_ants[phase],
            )

        # Instantiate environment if needed
        if env is None or isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # calculate heatmap
        heuristic_logp = self.heatmap_generator(graph)

        aco = self.aco_class(heuristic_logp, n_ants=self.n_ants[phase], **self.aco_args)
        td, actions, reward = aco.run(td_initial, env, self.n_iterations[phase])

        if calc_reward:
            td.set("reward", reward)

        return None, actions, td

    def _conv_params(self, value: Union[int, dict, None], **defaults):
        if value is None:
            return defaults
        elif isinstance(value, int):
            return {key: value for key in defaults.keys()}
        else:
            defaults.update(dict(**value))  # convert DictConfigs
            return defaults
