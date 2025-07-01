import math

from typing import Optional, Union

import numpy as np
import scipy
import torch

from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.models.zoo.deepaco import DeepACO
from rl4co.models.zoo.gfacs.policy import GFACSPolicy
from rl4co.utils.ops import unbatchify


class GFACS(DeepACO):
    """Implements GFACS: https://arxiv.org/abs/2403.07041.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline. Defaults to no baseline
        train_with_local_search: Whether to train with local search. Defaults to False.
        policy_kwargs: Keyword arguments for policy
        baseline_kwargs: Keyword arguments for baseline
        alpha_min: Minimum value for alpha. Defaults to 0.5.
        alpha_max: Maximum value for alpha. Defaults to 1.0.
        alpha_flat_epochs: Number of epochs to keep alpha constant. Defaults to 5.
        beta_min: Minimum value for beta. Defaults to 1.0.
        beta_max: Maximum value for beta. Defaults to 1.0.
        beta_flat_epochs: Number of epochs to keep beta constant. Defaults to 5.
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[GFACSPolicy] = None,
        baseline: Union[REINFORCEBaseline, str] = "no",
        train_with_local_search: bool = True,
        policy_kwargs: dict = {},
        baseline_kwargs: dict = {},
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        alpha_flat_epochs: int = 5,
        beta_min: float = 1.0,
        beta_max: float = 1.0,
        beta_flat_epochs: int = 5,
        **kwargs,
    ):
        if policy is None:
            policy = GFACSPolicy(
                env_name=env.name,
                train_with_local_search=train_with_local_search,
                **policy_kwargs,
            )

        super().__init__(
            env,
            policy,
            baseline,
            train_with_local_search,
            policy_kwargs=policy_kwargs,
            baseline_kwargs=baseline_kwargs,
            **kwargs,  # ls_reward_aug_W is replaced by \alpha in this paper
        )

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_flat_epochs = alpha_flat_epochs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_flat_epochs = beta_flat_epochs

    @property
    def alpha(self) -> float:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * min(
            self.current_epoch
            / (self.trainer.max_epochs - self.alpha_flat_epochs),  # type: ignore
            1.0,
        )

    @property
    def beta(self) -> float:
        return self.beta_min + (self.beta_max - self.beta_min) * min(
            math.log(self.current_epoch + 1)
            / math.log(self.trainer.max_epochs - self.beta_flat_epochs),  # type: ignore
            1.0,
        )

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss for REINFORCE algorithm.

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        reward = policy_out["reward"]
        n_ants = reward.size(1)  # type: ignore
        advantage = reward - reward.mean(dim=1, keepdim=True)  # type: ignore

        if self.train_with_local_search:
            ls_reward = policy_out["ls_reward"]
            ls_advantage = ls_reward - ls_reward.mean(dim=1, keepdim=True)
            weighted_advantage = advantage * (1 - self.alpha) + ls_advantage * self.alpha
        else:
            weighted_advantage = advantage

        # On-policy loss
        forward_flow = policy_out["log_likelihood"] + policy_out["logZ"].repeat(1, n_ants)
        backward_flow = (
            self.calculate_log_pb_uniform(policy_out["actions"], n_ants)
            + weighted_advantage.detach() * self.beta
        )
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()

        # Off-policy loss
        if self.train_with_local_search:
            ls_forward_flow = policy_out["ls_log_likelihood"] + policy_out[
                "ls_logZ"
            ].repeat(1, n_ants)
            ls_backward_flow = (
                self.calculate_log_pb_uniform(policy_out["ls_actions"], n_ants)
                + ls_advantage.detach() * self.beta
            )
            ls_tb_loss = torch.pow(ls_forward_flow - ls_backward_flow, 2).mean()
            tb_loss = tb_loss + ls_tb_loss

        return tb_loss

    def calculate_log_pb_uniform(self, actions: torch.Tensor, n_ants: int):
        match self.env.name:
            case "tsp":
                return math.log(1 / 2 * actions.size(1))
            case "cvrp":
                _a1 = actions.detach().cpu().numpy()
                # shape: (batch, max_tour_length)
                n_nodes = np.count_nonzero(_a1, axis=1)
                _a2 = _a1[:, 1:] - _a1[:, :-1]
                n_routes = np.count_nonzero(_a2, axis=1) - n_nodes
                _a3 = _a1[:, 2:] - _a1[:, :-2]
                n_multinode_routes = np.count_nonzero(_a3, axis=1) - n_nodes
                log_b_p = -scipy.special.gammaln(
                    n_routes + 1
                ) - n_multinode_routes * math.log(2)
                return unbatchify(torch.from_numpy(log_b_p).to(actions.device), n_ants)
            case "op" | "pctsp":
                return math.log(1 / 2)
            case _:
                raise ValueError(f"Unknown environment: {self.env.name}")
