from typing import Optional

import torch

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.dec_strategies import Sampling
from rl4co.models.zoo.common.nonautoregressive.decoder import (
    NonAutoregressiveDecoder as NARDecoder,
)


class AntSystem:
    """TODO"""

    def __init__(
        self,
        log_heuristic: torch.Tensor,
        n_ants: int = 20,
        n_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.95,
        pheromone: Optional[torch.Tensor] = None,
        require_logp: bool = False,
    ):
        self.batch_size, self.n_nodes, _ = log_heuristic.shape
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

        self.log_heuristic = log_heuristic
        self.pheromone = (
            torch.ones_like(log_heuristic) if pheromone is None else pheromone
        )

        self.final_actions = self.final_reward = None
        self.require_logp = require_logp
        self.all_records = []

        self._batchindex = torch.arange(self.batch_size, device=log_heuristic.device)
        self._batch_n_indices = (
            self._batchindex.unsqueeze(1).repeat(1, self.n_nodes).view(-1)
        )

    def run(
        self,
        td_initial: TensorDict,
        env: RL4COEnvBase,
        n_iterations: Optional[int] = None,
    ):
        n_iterations = n_iterations or self.n_iterations

        for _ in range(n_iterations):
            # reset environment
            td: TensorDict = env.reset(td_initial.clone(recurse=False))  # type: ignore
            reward, actions = self.one_step(td, env)

        return td, actions, reward

    def one_step(self, td: TensorDict, env: RL4COEnvBase):
        # sampling
        actions, reward = self._sampling(td, env)
        # local search, reserved for extensions
        actions, reward = self._local_search(actions, reward)
        # update final actions and rewards
        self._update_results(actions, reward)
        # update pheromone matrix
        self._update_pheromone(actions, reward)

        return reward, actions

    def _sampling(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
    ):
        # p = phe**alpha * heu**beta <==> log(p) = alpha*log(phe) + beta*log(heu)
        heatmaps_logp = (
            self.alpha * torch.log(self.pheromone) + self.beta * self.log_heuristic
        )
        self.decode_strategy = Sampling(multistart=True, num_starts=self.n_ants)
        td, env, num_starts = self.decode_strategy.pre_decoder_hook(td, env)
        while not td["done"].all():
            log_p, mask = NARDecoder._get_log_p(td, heatmaps_logp, num_starts)
            td = self.decode_strategy.step(log_p, mask, td)
            td = env.step(td)["next"]

        outputs, actions, td, env = self.decode_strategy.post_decoder_hook(td, env)
        reward = env.get_reward(td, actions)

        if self.require_logp:
            self.all_records.append((outputs, actions, reward))

        # reshape from (batch_size * n_ants, ...) to (batch_size, n_ants, ...)
        reward = reward.view(self.batch_size, self.n_ants)
        actions = actions.view(self.batch_size, self.n_ants, -1)

        return actions, reward

    def _local_search(self, actions, reward):
        # Override this method in childclass to perform local search.
        return actions, reward

    def _update_results(self, actions, reward):
        best_index = reward.argmax(-1)
        best_reward = reward[self._batchindex, best_index]
        best_actions = actions[self._batchindex, best_index]

        if self.final_actions is None or self.final_reward is None:
            self.final_actions = best_actions.clone()
            self.final_reward = best_reward.clone()
        else:
            require_update = self._batchindex[self.final_reward <= best_reward]
            self.final_actions[require_update, :] = best_actions[require_update]
            self.final_reward[require_update] = best_reward[require_update]

        return best_index

    def _update_pheromone(self, actions, reward):
        # calculate Δphe
        delta_pheromone = torch.zeros_like(self.pheromone)
        from_node = actions
        to_node = torch.roll(from_node, 1, -1)
        mapped_reward = self._reward_map(reward).detach()

        for ant_index in range(self.n_ants):
            delta_pheromone[
                self._batch_n_indices,
                from_node[:, ant_index].flatten(),
                to_node[:, ant_index].flatten(),
            ] += mapped_reward[self._batch_n_indices, ant_index]

        # decay & update
        self.pheromone *= self.decay
        self.pheromone += delta_pheromone

    def _reward_map(self, x):
        # map reward from $\mathbb{R}$ to $\mathbb{R}^+$
        return torch.where(x >= -2, 0.25 * x + 1, -1 / x)
