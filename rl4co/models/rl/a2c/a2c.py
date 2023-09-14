from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.a2c.baseline import CriticBaseline
from rl4co.models.rl.common.critic import CriticNetwork
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class A2C(REINFORCE):
    """A2C (Advantage Actor Critic) algorithm. A2C is based on policy gradients (REINFORCE)
    with a critic network as baseline, similarly to Kool et al. (2019).
    See superclass `REINFORCE` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy (actor) to use for the algorithm
        critic: Critic network to use as baseline. If None, create a new critic network based on the environment
        critic_kwargs: Keyword arguments for critic network. Ignored if critic is not None
        optimizer: Optimizer to use for the actor.
        optimizer_kwargs: Keyword arguments for optimizer
        optimizer_critic: Optimizer to use for the critic. If None, use the same optimizer as the actor
        optimizer_critic_kwargs: Keyword arguments for critic optimizer. Ignored if optimizer_critic is None
        gradient_clip_val: Clip gradients to this value. This has to be passed to the LightningModule instead of the
            Trainer, because of the manual optimization
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: nn.Module = None,
        critic_kwargs: dict = {},
        optimizer: Union[str, torch.optim.Optimizer, partial] = "Adam",
        optimizer_kwargs: dict = {"lr": 1e-4},
        optimizer_critic: Union[str, torch.optim.Optimizer, partial] = None,
        optimizer_critic_kwargs: dict = {"lr": 1e-4},
        gradient_clip_val: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, policy, **kwargs)

        self.save_hyperparameters(logger=False)

        if critic is None:
            critic = CriticNetwork(env.name, **critic_kwargs)

        # The critic becomes a baseline for the policy
        self.baseline = CriticBaseline(critic)

        self._optimizer_name_or_cls_actor: Union[str, torch.optim.Optimizer] = optimizer
        self.optimizer_kwargs_actor: dict = optimizer_kwargs

        if optimizer_critic is None:
            log.warning(
                "`optimizer_critic` not passed. Using the same optimizer for actor and critic"
            )
            optimizer_critic = optimizer
            optimizer_critic_kwargs = optimizer_kwargs

        self._optimizer_name_or_cls_critic: Union[
            str, torch.optim.Optimizer
        ] = optimizer_critic
        self.optimizer_kwargs_critic: dict = optimizer_critic_kwargs

        log.info(
            f"Using manual optimization with gradient clip value {gradient_clip_val}. Please pass the value to this class instead of the Trainer."
        )
        self.gradient_clip_val = gradient_clip_val
        self.automatic_optimization = False

    def configure_optimizers(self, parameters=None):
        """Configure all learning rate schedulers and optimizers

        Args:
            parameters: parameters to be optimized. If None, will use `self.policy.parameters()
        """

        actor_parameters, critic_parameters = (
            self.policy.parameters(),
            self.baseline.parameters(),
        )

        optimizers = []

        for name, params, opt, opt_kwargs in zip(
            ["actor", "critic"],
            [actor_parameters, critic_parameters],
            [self._optimizer_name_or_cls_actor, self._optimizer_name_or_cls_critic],
            [self.optimizer_kwargs_actor, self.optimizer_kwargs_critic],
        ):
            log.info(f"Instantiating optimizer <{opt}> for {name}")
            optimizers.append(self._configure_optimizer(params, opt, opt_kwargs))

        for opt in optimizers:
            log.info(f"Optimizer: {opt}")

        if self._lr_scheduler_name_or_cls is None:
            return optimizers
        else:
            schedulers = []
            log.info(f"Instantiating schedulers <{self._lr_scheduler_name_or_cls}>")
            for opt in optimizers:
                schedulers.append(
                    self._configure_lr_scheduler(
                        opt,
                        self._lr_scheduler_name_or_cls,
                        self.lr_scheduler_kwargs,
                        self.lr_scheduler_interval,
                        self.lr_scheduler_monitor,
                    )
                )
            return optimizers, schedulers

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
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # Critic baseline (similar to REINFORCE)
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        reinforce_loss = -(advantage * log_likelihood).mean()
        loss = reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )

        # Manually optimize
        self.manual_backward(loss)
        for opt in self.optimizers():
            if self.gradient_clip_val is not None:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
            opt.step()
            opt.zero_grad()

        # Scheduler if per step
        if self.lr_scheduler_interval == "step":
            for scheduler in self.lr_schedulers():
                scheduler.step()

        return policy_out
