from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from rl4co.utils.pylogger import get_pylogger
from sympy import N
from tensordict import TensorDict

log = get_pylogger(__name__)


class PPO(nn.Module):
    def __init__(
        self,
        env,
        policy: nn.Module,
        critic: nn.Module,
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 1,  # K
        mini_batch_size: int = 16,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = None,  # max gradient norm
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.critic = critic

        # PPO hyper params
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.vf_lambda = vf_lambda
        self.entropy_lambda = entropy_lambda
        self.normalize_adv = normalize_adv
        self.max_grad_norm = max_grad_norm

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        extra=None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
        optimizer=None,
    ):
        # Evaluate model, get costs and log probabilities
        with torch.no_grad():
            # compute a_old and logp_old
            out = self.policy(td, phase, return_action=True, **policy_kwargs)
            old_logp = out["log_likelihood"]  # [batch, decoder steps]
            actions = out["actions"]  # [batch, decoder steps]
            rewards = out["reward"]  # [batch]

            old_values = self.critic(td, **critic_kwargs)

        if phase == "train":
            for _ in range(self.ppo_epochs):  # loop K
                batch_size = old_logp.shape[0]

                if self.mini_batch_size > batch_size:
                    log.info(
                        "Mini batch size is larger than batch size, set to batch size"
                    )
                    mini_batch_size = batch_size
                else:
                    mini_batch_size = self.mini_batch_size

                iter_i = 0
                for mini_batch_idx in torch.randperm(batch_size).split(mini_batch_size):
                    mini_batched_td = td[mini_batch_idx]
                    # print("mini_batched_td", mini_batched_td["action_mask"].sum())

                    # compute a and logp
                    mini_batched_out = self.policy(
                        mini_batched_td,
                        phase,
                        given_actions=actions[mini_batch_idx],
                        return_entropy=True,
                        calc_reward=False,
                        **policy_kwargs
                    )

                    # compute ratio
                    ratio = torch.exp(
                        mini_batched_out["selected_log_p"] - old_logp[mini_batch_idx]
                    ).sum(
                        dim=-1
                    )  # [batch size]

                    # compute advantage
                    adv = rewards[mini_batch_idx] - old_values[mini_batch_idx].view(
                        -1
                    )  # [batch size]

                    if self.normalize_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

                    # compute surrogate loss
                    surrogate_loss = torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        * adv,
                    ).mean()

                    # compute entropy bonus
                    entropy_bonus = mini_batched_out["entropy"].mean()

                    # compute value function loss
                    value = self.critic(mini_batched_td, **critic_kwargs)
                    value_loss = F.huber_loss(
                        value, rewards[mini_batch_idx].view(-1, 1)
                    )
                    # value_loss = (value - rewards[mini_batch_idx]).pow(2).mean()

                    # compute total loss
                    loss = (
                        -surrogate_loss
                        + self.vf_lambda * value_loss
                        # - self.entropy_lambda * entropy_bonus
                    )

                    # perform optimization
                    if optimizer is not None:
                        optimizer.zero_grad()
                        loss.backward()
                        if self.max_grad_norm is not None:
                            nn.utils.clip_grad_norm_(
                                self.parameters(), self.max_grad_norm
                            )
                        optimizer.step()

            # log training results
            out.update(
                {
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    "value_loss": value_loss,
                    "entropy_bonus": entropy_bonus,
                }
            )
        return out
