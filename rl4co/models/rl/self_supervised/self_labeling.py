import copy

from typing import Any, Union

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    ListStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
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


class SelfLabeling(RL4COLitModule):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        clip_range: float = 0.2,  # epsilon of PPO
        update_timestep: int = 1,
        buffer_size: int = 100_000,
        sl_epochs: int = 1,  # inner epoch, K
        batch_size: int = 256,
        mini_batch_size: int = 256,
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.01,  # lambda of entropy bonus
        max_grad_norm: float = 0.5,  # max gradient norm
        buffer_storage_device: str = "gpu",
        metrics: dict = {
            "train": ["loss", "surrogate_loss", "value_loss", "entropy"],
        },
        reward_scale: Union[str, int] = None,
        num_starts: int = None,
        **kwargs,
    ):
        super().__init__(env, policy, metrics=metrics, batch_size=batch_size, **kwargs)

        self.policy_old = copy.deepcopy(self.policy)
        self.automatic_optimization = False  # PPO uses custom optimization routine
        self.rb = make_replay_buffer(buffer_size, mini_batch_size, buffer_storage_device)
        self.sl_epochs = sl_epochs
        self.max_grad_norm = max_grad_norm
        self.update_timestep = update_timestep
        self.mini_batch_size = mini_batch_size
        self.num_starts = num_starts

    def update(self, eval_td, device):
        losses = []
        # PPO inner epoch
        for _ in range(self.sl_epochs):
            for sub_td in self.rb:
                sub_td = sub_td.to(device)

                logprobs, _, _ = self.policy.evaluate(sub_td, return_selected=False)

                criterion = CrossEntropyLoss(reduction="mean")
                # compute total loss
                loss = criterion(logprobs, sub_td["action"])

                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.max_grad_norm is not None:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.max_grad_norm,
                        gradient_clip_algorithm="norm",
                    )

                opt.step()
                losses.append(loss)

        # need eval for greedy decoding
        out = self.policy.generate(eval_td, self.env, phase="val")
        # add loss to metrics
        out["loss"] = torch.stack(losses, dim=0)
        return out

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        orig_td = self.env.reset(batch)
        device = orig_td.device
        n_start = (
            self.env.get_num_starts(orig_td)
            if self.num_starts is None
            else self.num_starts
        )
        next_td = batchify(orig_td.clone(), n_start)
        td_stack = []

        if phase == "train":
            while not next_td["done"].all():

                with torch.no_grad():
                    td = self.policy_old.act(next_td, self.env, phase="train")

                # get next state
                next_td = self.env.step(td)["next"]

                # add tensordict with action, logprobs and reward information to buffer
                td_stack.append(td)
            # (bs * #samples, #steps)
            td_stack = torch.stack(td_stack, dim=1)
            # (bs, #samples, #steps)
            td_stack_unbs = unbatchify(td_stack, n_start)
            # (bs * #samples)
            rewards = self.env.get_reward(next_td, None)
            # (bs)
            _, best_idx = unbatchify(rewards, n_start).max(dim=1)
            td_best = td_stack_unbs.gather(
                1, best_idx[:, None, None].expand(-1, 1, td_stack_unbs.size(2))
            ).squeeze(1)
            # flatten so that every step is an experience TODO can we enhance this?
            self.rb.extend(td_best.flatten())

            # if iter mod x = 0 then update the policy (x = 1 in paper)
            if batch_idx % self.update_timestep == 0:

                out = self.update(orig_td, device)

                # TODO check the details of this: if out["reward"].mean() > max_rew.mean():
                # Copy new weights into old policy:
                self.policy_old.load_state_dict(self.policy.state_dict())
                # only clear the rb if we improved on the old model, otherwise the experience is still useful
                self.rb.empty()

        else:
            out = self.policy.generate(
                next_td, self.env, phase=phase  # , select_best=True, multisample=True
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
