from typing import Optional, Union

import torch
import torch.nn as nn

from tensordict import TensorDict

try:
    from torch_geometric.data import Batch
except ImportError:
    # `Batch` is referred to only as type notations in this file
    Batch = None

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.dec_strategies import Sampling
from rl4co.models.zoo.common.nonautoregressive.decoder import NonAutoregressiveDecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DeepACODecoder(NonAutoregressiveDecoder):
    """TODO"""

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int,
        num_layers: int,
        heatmap_generator: Optional[nn.Module] = None,
        linear_bias: bool = True,
        n_ants: int = 20,
        n_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.95,
    ) -> None:
        super(DeepACODecoder, self).__init__(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            heatmap_generator=heatmap_generator,
            linear_bias=linear_bias,
        )
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

    def forward(
        self,
        td_initial: TensorDict,
        graph: Batch,
        env: Union[str, RL4COEnvBase, None] = None,
        calc_reward: bool = True,
        n_ants: Optional[int] = None,
        n_iterations: Optional[int] = None,
        phase="train",
        **unused_kwargs,
    ):
        """TODO"""
        if phase == "train":
            # use procedure inherited from NonAutoregressiveDecoder
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

        n_ants = n_ants or self.n_ants
        n_iterations = n_iterations or self.n_iterations
        # batchsize = heuristic_logp.shape[0]

        pheromone = torch.ones_like(heuristic_logp)

        for _ in range(self.n_iterations):
            td = env.reset(td_initial.clone(recurse=False))
            heatmaps_logp = self._aco_get_heatmaps_logp(pheromone, heuristic_logp)
            outputs, actions, env, td, reward = self._aco_sampling(
                td, env, heatmaps_logp, n_ants
            )
            # TODO: complete ACO main loop

        td.set("reward", reward)

        return outputs, actions, td

    def _aco_sampling(
        self, td: TensorDict, env: RL4COEnvBase, heatmaps_logp: torch.Tensor, n_ants: int
    ):
        self.decode_strategy = Sampling(multistart=True, num_starts=n_ants)
        td, env, num_starts = self.decode_strategy.pre_decoder_hook(td, env)
        while not td["done"].all():
            log_p, mask = self._get_log_p(td, heatmaps_logp, num_starts)
            td = self.decode_strategy.step(log_p, mask, td)
            td = env.step(td)["next"]

        outputs, actions, td, env = self.decode_strategy.post_decoder_hook(td, env)
        reward = env.get_reward(td, actions)

        return outputs, actions, env, td, reward

    def _aco_get_heatmaps_logp(
        self, pheromone: torch.Tensor, log_heuristic: torch.Tensor
    ):
        # p = phe**alpha * heu**beta <==> log(p) = alpha*log(phe) + beta*log(heu)
        return self.alpha * torch.log(pheromone) + self.beta * log_heuristic
