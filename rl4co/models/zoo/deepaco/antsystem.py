from functools import lru_cache
from typing import Optional, Tuple

import torch

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.dec_strategies import Sampling
from rl4co.models.zoo.common.nonautoregressive.decoder import (
    NonAutoregressiveDecoder as NARDecoder,
)


class AntSystem:
    """Implements the Ant System algorithm: https://doi.org/10.1109/3477.484436.

    Args:
        log_heuristic: Logarithm of the heuristic matrix.
        n_ants: Number of ants to be used in the algorithm. Defaults to 20.
        alpha: Importance of pheromone in the decision-making process. Defaults to 1.0.
        beta: Importance of heuristic information in the decision-making process. Defaults to 1.0.
        decay: Rate at which pheromone evaporates. Should be between 0 and 1. Defaults to 0.95.
        pheromone: Initial pheromone matrix. Defaults to `torch.ones_like(log_heuristic)`.
        require_logp: Whether to require the log probability of actions. Defaults to False.
    """

    def __init__(
        self,
        log_heuristic: Tensor,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 1.0,
        decay: float = 0.95,
        pheromone: Optional[Tensor] = None,
        require_logp: bool = False,
    ):
        self.batch_size = log_heuristic.shape[0]
        self.n_ants = n_ants
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

    def run(
        self,
        td_initial: TensorDict,
        env: RL4COEnvBase,
        n_iterations: int,
    ) -> Tuple[TensorDict, Tensor, Tensor]:
        """Run the Ant System algorithm for a specified number of iterations.

        Args:
            td_initial: Initial state of the problem.
            env: Environment representing the problem.
            n_iterations: Number of iterations to run the algorithm.

        Returns:
            td: The final state of the problem.
            actions: The final actions chosen by the algorithm.
            reward: The final reward achieved by the algorithm.
        """
        for _ in range(n_iterations):
            # reset environment
            td: TensorDict = env.reset(td_initial.clone(recurse=False))  # type: ignore
            self.one_step(td, env)

        td, env = self._recreate_final_routes(td_initial, env)

        assert self.final_actions is not None and self.final_reward is not None
        return td, self.final_actions, self.final_reward

    def one_step(self, td: TensorDict, env: RL4COEnvBase):
        """Run one step of the Ant System algorithm.

        Args:
            td: Current state of the problem.
            env: Environment representing the problem.

        Returns:
            actions: The actions chosen by the algorithm.
            reward: The reward achieved by the algorithm.
        """
        # sampling
        actions, reward = self._sampling(td, env)
        # local search, reserved for extensions
        actions, reward = self._local_search(actions, reward)
        # update final actions and rewards
        self._update_results(actions, reward)
        # update pheromone matrix
        self._update_pheromone(actions, reward)

        return actions, reward

    def _sampling(
        self,
        td: TensorDict,
        env: RL4COEnvBase,
    ):
        # Sample from heatmaps
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
            self.all_records.append((outputs, actions, reward, td.get("mask", None)))

        # reshape from (batch_size * n_ants, ...) to (batch_size, n_ants, ...)
        reward = reward.view(self.batch_size, self.n_ants)
        actions = actions.view(self.batch_size, self.n_ants, -1)

        return actions, reward

    def _local_search(self, actions: Tensor, reward: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform local search on the actions and reward obtained.

        This method is reserved for extensions and can be overridden in child classes to implement
        specific local search strategies. By default, it returns the original actions and reward
        without any modifications.

        Args:
            actions: Actions chosen by the algorithm.
            reward: Reward obtained after the sampling step.

        Returns:
            actions: The modified actions
            reward: The modified reward

        """
        return actions, reward

    def _update_results(self, actions, reward):
        # update the best-trails recorded in self.final_actions
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
        batch_action_indices = self._batch_action_indices(
            self.batch_size, actions.shape[-1], reward.device
        )

        for ant_index in range(self.n_ants):
            delta_pheromone[
                batch_action_indices,
                from_node[:, ant_index].flatten(),
                to_node[:, ant_index].flatten(),
            ] += mapped_reward[batch_action_indices, ant_index]

        # decay & update
        self.pheromone *= self.decay
        self.pheromone += delta_pheromone

    def _reward_map(self, x):
        # map reward from $\mathbb{R}$ to $\mathbb{R}^+$
        return torch.where(x >= -2, 0.25 * x + 1, -1 / x)

    def _recreate_final_routes(self, td, env):
        assert self.final_actions is not None

        for action_index in range(self.final_actions.shape[-1]):
            actions = self.final_actions[:, action_index]
            td.set("action", actions)
            td = env.step(td)["next"]

        assert td["done"].all()
        return td, env

    def get_logp(self):
        """Get the log probability (logp) values recorded during the execution of the algorithm.

        Returns:
            results: Tuple containing the log probability values,
                actions chosen, rewards obtained, and mask values (if available).

        Raises:
            AssertionError: If `require_logp` is not enabled.
        """

        assert self.require_logp, "Please enable `require_logp` to record logp values"

        logp_list, actions_list, reward_list, mask_list = [], [], [], []

        for outputs, actions, reward, mask in self.all_records:
            logp_list.append(outputs)
            actions_list.append(actions)
            reward_list.append(reward)
            mask_list.append(mask)

        if mask_list[0] is None:
            mask_list = None
        else:
            mask_list = torch.stack(mask_list, 0)

        # reset records
        self.all_records = []

        return (
            torch.stack(logp_list, 0),
            torch.stack(actions_list, 0),
            torch.stack(reward_list, 0),
            mask_list,
        )

    @staticmethod
    @lru_cache(5)
    def _batch_action_indices(batch_size: int, n_actions: int, device: torch.device):
        batchindex = torch.arange(batch_size, device=device)
        return batchindex.unsqueeze(1).repeat(1, n_actions).view(-1)
