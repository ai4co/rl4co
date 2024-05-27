from typing import Iterable, List, Optional, Union

import lightning.pytorch as pl
import torch
import math
import copy
from torch.optim import Adam

from lightning import Callback, Trainer
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from rl4co import utils
import random
log = utils.get_pylogger(__name__)


class MetaModelCallback(Callback):
    def __init__(self, meta_params, print_log=True):
        super().__init__()
        self.meta_params = meta_params
        assert meta_params["meta_method"] == 'reptile', NotImplementedError
        assert meta_params["data_type"] == 'size', NotImplementedError
        self.print_log = print_log

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        # Initialize some hyperparameters
        self.alpha = self.meta_params["alpha"]
        self.alpha_decay = self.meta_params["alpha_decay"]
        self.sch_bar = self.meta_params["sch_bar"]
        self.task_set = [(n,) for n in range(self.meta_params["min_size"], self.meta_params["max_size"] + 1)]

        # Sample a batch of tasks
        self._sample_task()
        self.selected_tasks[0] = (pl_module.env.generator.num_loc, )

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        # Alpha scheduler (decay for the update of meta model)
        self._alpha_scheduler()

        # Reinitialize the task model with the parameters of the meta model
        if trainer.current_epoch %  self.meta_params['B'] == 0: # Save the meta model
            self.meta_model_state_dict = copy.deepcopy(pl_module.state_dict())
            self.task_models = []
            # Print sampled tasks
            if self.print_log:
                print('\n>> Meta epoch: {} (Exact epoch: {}), Training task: {}'.format(trainer.current_epoch//self.meta_params['B'], trainer.current_epoch, self.selected_tasks))
        else:
            pl_module.load_state_dict(self.meta_model_state_dict)

        # Reinitialize the optimizer every epoch
        lr_decay = 0.1 if trainer.current_epoch+1 == int(self.sch_bar * trainer.max_epochs) else 1
        old_lr  = trainer.optimizers[0].param_groups[0]['lr']
        new_optimizer = Adam(pl_module.parameters(), lr=old_lr * lr_decay)
        trainer.optimizers = [new_optimizer]

        # Print
        if self.print_log:
            print('\n>> Training task: {}, capacity: {}'.format(pl_module.env.generator.num_loc, pl_module.env.generator.capacity))

    def on_train_epoch_end(self, trainer, pl_module):

        # Save the task model
        self.task_models.append(copy.deepcopy(pl_module.state_dict()))
        if (trainer.current_epoch+1) % self.meta_params['B'] == 0:
            # Outer-loop optimization (update the meta model with the parameters of the task model)
            with torch.no_grad():
                state_dict = {params_key: (self.meta_model_state_dict[params_key] +
                                           self.alpha * torch.mean(torch.stack([fast_weight[params_key] - self.meta_model_state_dict[params_key]
                                                                                for fast_weight in self.task_models], dim=0).float(), dim=0))
                              for params_key in self.meta_model_state_dict}
                pl_module.load_state_dict(state_dict)

        # Get ready for the next meta-training iteration
        if (trainer.current_epoch + 1) % self.meta_params['B'] == 0:
            # Sample a batch of tasks
            self._sample_task()

        # Load new training task (Update the environment)
        self._load_task(pl_module, task_idx = (trainer.current_epoch+1) % self.meta_params['B'])

    def _sample_task(self):
        # Sample a batch of tasks
        w, self.selected_tasks = [1.0] * self.meta_params['B'], []
        for b in range(self.meta_params['B']):
            task_params = random.sample(self.task_set, 1)[0]
            self.selected_tasks.append(task_params)
        self.w = torch.softmax(torch.Tensor(w), dim=0)

    def _load_task(self, pl_module, task_idx=0):
        # Load new training task (Update the environment)
        task_params, task_w = self.selected_tasks[task_idx], self.w[task_idx].item()
        task_capacity = math.ceil(30 + task_params[0] / 5) if task_params[0] >= 20 else 20
        pl_module.env.generator.num_loc = task_params[0]
        pl_module.env.generator.capacity = task_capacity

    def _alpha_scheduler(self):
        self.alpha = max(self.alpha * self.alpha_decay, 0.0001)

class RL4COMetaTrainer(Trainer):
    """Wrapper around Lightning Trainer, with some RL4CO magic for efficient training.

    # Meta training framework for addressing the generalization issue
    # Based on Zhou et al. (2023): https://arxiv.org/abs/2305.19587

    Note:
        The most important hyperparameter to use is `reload_dataloaders_every_n_epochs`.
        This allows for datasets to be re-created on the run and distributed by Lightning across
        devices on each epoch. Setting to a value different than 1 may lead to overfitting to a
        specific (such as the initial) data distribution.

    Args:
        accelerator: hardware accelerator to use.
        callbacks: list of callbacks.
        logger: logger (or iterable collection of loggers) for experiment tracking.
        min_epochs: minimum number of training epochs.
        max_epochs: maximum number of training epochs.
        strategy: training strategy to use (if any), such as Distributed Data Parallel (DDP).
        devices: number of devices to train on (int) or which GPUs to train on (list or str) applied per node.
        gradient_clip_val: 0 means don't clip. Defaults to 1.0 for stability.
        precision: allows for mixed precision training. Can be specified as a string (e.g., '16').
            This also allows to use `FlashAttention` by default.
        disable_profiling_executor: Disable JIT profiling executor. This reduces memory and increases speed.
        auto_configure_ddp: Automatically configure DDP strategy if multiple GPUs are available.
        reload_dataloaders_every_n_epochs: Set to a value different than 1 to reload dataloaders every n epochs.
        matmul_precision: Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        **kwargs: Additional keyword arguments passed to the Lightning Trainer. See :class:`lightning.pytorch.trainer.Trainer` for details.
    """

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Union[Logger, Iterable[Logger]]] = None,
        min_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        gradient_clip_val: Union[int, float] = 1.0,
        precision: Union[str, int] = "16-mixed",
        reload_dataloaders_every_n_epochs: int = 1,
        disable_profiling_executor: bool = True,
        auto_configure_ddp: bool = True,
        matmul_precision: Union[str, int] = "medium",
        **kwargs,
    ):
        # Disable JIT profiling executor. This reduces memory and increases speed.
        # Reference: https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
        if disable_profiling_executor:
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass

        # Configure DDP automatically if multiple GPUs are available
        if auto_configure_ddp and strategy == "auto":
            if devices == "auto":
                n_devices = num_cuda_devices()
            elif isinstance(devices, Iterable):
                n_devices = len(devices)
            else:
                n_devices = devices
            if n_devices > 1:
                log.info(
                    "Configuring DDP strategy automatically with {} GPUs".format(
                        n_devices
                    )
                )
                strategy = DDPStrategy(
                    find_unused_parameters=True,  # We set to True due to RL envs
                    gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
                )

        # Set matmul precision for faster inference https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)

        # Check if gradient_clip_val is set to None
        if gradient_clip_val is None:
            log.warning(
                "gradient_clip_val is set to None. This may lead to unstable training."
            )

        # We should reload dataloaders every epoch for RL training
        if reload_dataloaders_every_n_epochs != 1:
            log.warning(
                "We reload dataloaders every epoch for RL training. Setting reload_dataloaders_every_n_epochs to a value different than 1 "
                + "may lead to unexpected behavior since the initial conditions will be the same for `n_epochs` epochs."
            )

        # Main call to `Trainer` superclass
        super().__init__(
            accelerator=accelerator,
            callbacks=callbacks,
            logger=logger,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            strategy=strategy,
            gradient_clip_val=gradient_clip_val,
            devices=devices,
            precision=precision,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            **kwargs,
        )

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """
        We override the `fit` method to automatically apply and handle RL4CO magic
        to 'self.automatic_optimization = False' models, such as PPO

        It behaves exactly like the original `fit` method, but with the following changes:
        - if the given model is 'self.automatic_optimization = False', we override 'gradient_clip_val' as None
        """

        if not model.automatic_optimization:
            if self.gradient_clip_val is not None:
                log.warning(
                    "Overriding gradient_clip_val to None for 'automatic_optimization=False' models"
                )
                self.gradient_clip_val = None

        # Fit (Inner-loop Optimization)
        super().fit(
            model=model,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )



