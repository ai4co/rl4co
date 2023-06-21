from math import log
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PPO(nn.Module):
    def __init__(
        self,
        env,
        policy: nn.Module,
        critic: nn.Module,
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 2,  # K
        mini_batch_size: Union[int, float] = 0.25,  # 0.25,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        normalize_adv: bool = False,  # whether to normalize advantage
        max_grad_norm: float = 0.5,  # max gradient norm
        **unused_kw,
    ):
        super().__init__()
        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")
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
            out = self.policy(td.clone(), phase, return_action=True, **policy_kwargs)
            old_logp = out["log_likelihood"]  # [batch, decoder steps]
            actions = out["actions"]  # [batch, decoder steps]
            rewards = out["reward"]  # [batch]

        iter_i = 0
        if phase == "train":
            batch_size = old_logp.shape[0]

            if isinstance(self.mini_batch_size, float):
                mini_batch_size = int(self.mini_batch_size * batch_size)
            if self.mini_batch_size >= batch_size:
                mini_batch_size = batch_size

            for _ in range(self.ppo_epochs):  # loop K
                for mini_batch_idx in torch.randperm(batch_size).split(mini_batch_size):
                    # compute a and logp
                    mini_batched_out = self.policy(
                        td[mini_batch_idx].clone(),
                        phase,
                        given_actions=actions[mini_batch_idx],
                        return_entropy=True,
                        calc_reward=False,
                        **policy_kwargs,
                    )

                    # compute ratio
                    ratio = torch.exp(
                        mini_batched_out["selected_log_p"].sum(dim=-1)
                        - old_logp[mini_batch_idx].sum(dim=-1)
                    )  # [batch size]

                    # compute advantage

                    value_pred = self.critic(td[mini_batch_idx], **critic_kwargs)
                    adv = rewards[mini_batch_idx] - value_pred.detach()  # [batch size]

                    if self.normalize_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

                    # compute surrogate loss
                    surrogate_loss = -torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        * adv,
                    ).mean()

                    # compute entropy bonus
                    entropy_bonus = mini_batched_out["entropy"].mean()

                    # compute value function loss
                    value_loss = F.huber_loss(
                        value_pred, rewards[mini_batch_idx].view(-1, 1)
                    )

                    # compute total loss
                    loss = (
                        surrogate_loss
                        + self.vf_lambda * value_loss
                        - self.entropy_lambda * entropy_bonus
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

                        iter_i += 1

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
