import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.data import DataLoader

from rl4co.models.rl.reinforce.base import REINFORCE


class PPO(REINFORCE):
    def __init__(
        self,
        env,
        policy: nn.Module,
        critic: nn.Module,
        clip_range: float = 0.2,  # epsilon of PPO
        ppo_epochs: int = 2,  # K
        mini_batch_size: int = 32,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
    ):
        super(PPO, self).__init__(env=env)
        self.policy = policy
        self.critic = critic

        # PPO hyper params
        self.clip_range = clip_range
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.vf_lambda = vf_lambda
        self.entropy_lambda = entropy_lambda

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        extra=None,
        policy_kwargs: dict = {},
        critic_kwargs: dict = {},
    ):
        # Evaluate model, get costs and log probabilities
        with torch.no_grad():
            # compute a_old and logp_old
            out = self.policy(td, phase, **policy_kwargs)

        if phase == "train":
            # Construct Datalaoder
            data_loader = DataLoader(
                td, batch_size=self.mini_batch_size, shuffle=True, collate_fn=lambda x: x
            )

            for _ in range(self.ppo_epochs):  # loop K
                for mini_batched_td in data_loader:
                    # compute a and logp
                    mini_batched_out = self.policy(mini_batched_td, phase, **policy_kwargs)

                    # compute ratio
                    ratio = torch.exp(mini_batched_out["log_likelihood"] - out["log_likelihood"])

                    # compute surrogate loss
                    surrogate_loss = torch.min(
                        ratio * mini_batched_out["reward"],
                        torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        * mini_batched_out["reward"],
                    ).mean()

                    # compute entropy bonus
                    entropy_bonus = mini_batched_out["entropy"].mean()

                    # compute value function loss
                    value = self.critic(mini_batched_td, phase, **critic_kwargs)["value"]
                    value_loss = (value - mini_batched_out["reward"]).pow(2).mean()

                    # compute total loss
                    loss = (
                        -surrogate_loss
                        + self.vf_lambda * value_loss
                        - self.entropy_lambda * entropy_bonus
                    )

                    # update
                    self.policy.optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    loss.backward()
                    self.policy.optimizer.step()
                    self.critic.optimizer.step()
