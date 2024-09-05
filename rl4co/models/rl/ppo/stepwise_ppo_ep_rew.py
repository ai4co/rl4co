import copy

from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    ListStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.utils import RewardScaler
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def make_replay_buffer(buffer_size, batch_size, device="cpu"):
    if device == "cpu":
        storage = LazyMemmapStorage(buffer_size, device="cpu")
        prefetch = 3
    else:
        storage = ListStorage(buffer_size)
        prefetch = None
    return TensorDictReplayBuffer(
        storage=storage,
        batch_size=batch_size,
        sampler=SamplerWithoutReplacement(drop_last=True),
        pin_memory=False,
        prefetch=prefetch,
    )


class StepwisePPOwEpisodicReward(RL4COLitModule):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        n_start: int = 0,
        clip_range: float = 0.2,  # epsilon of PPO
        buffer_size: int = 100_000,
        ppo_epochs: int = 2,  # inner epoch, K
        batch_size: int = 256,
        mini_batch_size: int = 256,
        rollout_batch_size: int = 256,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        max_grad_norm: float = 1.0,  # max gradient norm
        buffer_storage_device: str = "gpu",
        metrics: dict = {
            "train": ["loss", "surrogate_loss", "value_loss", "entropy"],
        },
        reward_scale: Union[str, int] = None,
        **kwargs,
    ):
        super().__init__(env, policy, metrics=metrics, batch_size=batch_size, **kwargs)

        self.policy_old = copy.deepcopy(self.policy)
        self.automatic_optimization = False  # PPO uses custom optimization routine
        self.rb = make_replay_buffer(buffer_size, mini_batch_size, buffer_storage_device)
        self.scaler = RewardScaler(reward_scale)
        self.n_start = n_start
        self.rollout_batch_size = rollout_batch_size
        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": mini_batch_size,
            "vf_lambda": vf_lambda,
            "entropy_lambda": entropy_lambda,
            "max_grad_norm": max_grad_norm,
        }

    def update(self, device):
        outs = []
        # PPO inner epoch
        for _ in range(self.ppo_cfg["ppo_epochs"]):
            for sub_td in self.rb:
                sub_td = sub_td.to(device)
                previous_reward = sub_td["reward"].view(-1, 1)
                previous_logp = sub_td["logprobs"]

                logprobs, value_pred, entropy = self.policy.evaluate(sub_td)

                ratios = torch.exp(logprobs - previous_logp)
                try:
                    advantages = sub_td["advantage"]
                    value_loss = 0

                except KeyError:
                    advantages = torch.squeeze(previous_reward - value_pred.detach(), 1)
                    # compute value function loss
                    value_loss = F.mse_loss(value_pred, previous_reward)
                surr1 = ratios * advantages
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.ppo_cfg["clip_range"],
                        1 + self.ppo_cfg["clip_range"],
                    )
                    * advantages
                )
                surrogate_loss = -torch.min(surr1, surr2).mean()

                # compute total loss
                loss = (
                    surrogate_loss
                    + self.ppo_cfg["vf_lambda"] * value_loss
                    - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                )

                # perform manual optimization following the Lightning routine
                # https://lightning.ai/docs/pytorch/stable/common/optimization.html

                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.ppo_cfg["max_grad_norm"] is not None:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                        gradient_clip_algorithm="norm",
                    )
                opt.step()

                out = {
                    "reward": previous_reward.mean(),
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    # "value_loss": value_loss,
                    "entropy": entropy.mean(),
                }

                outs.append(out)
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        outs = {k: torch.stack([dic[k] for dic in outs], dim=0) for k in outs[0]}
        return outs

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):

        if phase == "train":

            for i in range(0, batch.shape[0], self.rollout_batch_size):

                mini_batch = batch[i : i + self.rollout_batch_size]
                rollout_td_buffer = []
                next_td = self.env.reset(mini_batch)
                device = next_td.device

                if self.n_start > 1:
                    next_td = batchify(next_td, self.n_start)

                n_steps = 0
                while not next_td["done"].all():

                    with torch.no_grad():
                        td = self.policy_old.act(
                            next_td, self.env, phase="train", temp=2.0
                        )

                    rollout_td_buffer.append(td)
                    # get next state
                    next_td = self.env.step(td)["next"]
                    n_steps += 1

                # get rewards
                reward = self.env.get_reward(next_td, 1) / n_steps

                if self.n_start > 1:
                    reward_unbatched = unbatchify(reward, self.n_start)
                    advantage = (
                        reward
                        - batchify(reward_unbatched.mean(-1), self.n_start).detach()
                    )
                    advantage = self.scaler(advantage)
                    rollout_td_buffer = [
                        td.set("advantage", advantage) for td in rollout_td_buffer
                    ]

                else:
                    reward = self.scaler(reward)
                    rollout_td_buffer = [
                        td.set("reward", reward) for td in rollout_td_buffer
                    ]

                # add tensordict with action, logprobs and reward information to buffer
                self.rb.extend(torch.cat(rollout_td_buffer, dim=0))

            # if iter mod x = 0 then update the policy (x = 1 in paper)
            out = self.update(device)

            self.rb.empty()
            torch.cuda.empty_cache()

        else:
            next_td = self.env.reset(batch)
            out = self.policy.generate(
                next_td, self.env, phase=phase, select_best=phase != "train"
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)

        return {"loss": out.get("loss", None), **metrics}
