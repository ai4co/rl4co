from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader

from rl4co.data.dataset import tensordict_collate_fn
from rl4co.envs.base import EnvBase
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class RL4COLitModule(LightningModule):
    """
    Base LightningModule for Neural Combinatorial Optimization
    Args:
        cfg: Hydra config
        env: Environment to use overridding the config. If None, instantiate from config
        model: Model to use overridding the config. If None, instantiate from config
    """

    def __init__(self, cfg: DictConfig, env: EnvBase = None, model: nn.Module = None):
        if cfg.get("train", {}).get("disable_profiling", True):
            # Disable profiling executor. This reduces memory and increases speed.
            # https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass

        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        cfg = DictConfig(cfg) if not isinstance(cfg, DictConfig) else cfg
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Instantiate environment, model and metrics
        self.env = env if env is not None else self.instantiate_env()
        self.model = model if model is not None else self.instantiate_model()
        self.instantiate_metrics()

    def instantiate_env(self):
        log.info(f"Instantiating environment <{self.cfg.env._target_}>")
        return instantiate(self.cfg.env)

    def instantiate_model(self):
        log.info(f"Instantiating model <{self.cfg.model._target_}>")
        return instantiate(self.cfg.model, env=self.env)

    def instantiate_metrics(self):
        """Dictionary of metrics to be logged at each phase"""
        metrics = self.cfg.get("metrics", {})
        if not metrics:
            log.info(f"No metrics specified, using default")
        self.train_metrics = metrics.get("train", ["loss", "reward"])
        self.val_metrics = metrics.get("val", ["reward"])
        self.test_metrics = metrics.get("test", ["reward"])
        self.log_on_step = metrics.get("log_on_step", True)

    def setup(self, stage="fit"):
        # setup batch sizes
        if self.cfg.data.get("train_batch_size", None) is None:
            batch_size = self.cfg.data.get("batch_size", None)
            if batch_size is not None:
                log.warning(
                    f"batch_size under data specified, using default as {batch_size}"
                )
                self.cfg.data.train_batch_size = batch_size
            log.warning(
                f"No train_batch_size under data specified, using default as 64"
            )
        self.train_batch_size = self.cfg.data.get("train_batch_size", 64)
        self.val_batch_size = self.cfg.data.get("val_batch_size", self.train_batch_size)
        self.test_batch_size = self.cfg.data.get(
            "test_batch_size", self.train_batch_size
        )

        log.info(f"Setting up datasets")
        self.train_dataset = self.wrap_dataset(
            self.env.dataset(self.cfg.data.train_size, "train")
        )
        self.val_dataset = self.env.dataset(self.cfg.data.val_size, "val")
        test_size = self.cfg.data.get("test_size", self.cfg.data.val_size)
        self.test_dataset = self.env.dataset(test_size, "test")
        if hasattr(self.model, "setup"):
            self.model.setup(self)

    def configure_optimizers(self):
        train_cfg = self.cfg.get("train", {})
        if train_cfg.get("optimizer", None) is None:
            log.info(f"No optimizer specified, using default")
        opt_cfg = train_cfg.get(
            "optimizer", DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-4})
        )
        if "_target_" not in opt_cfg:
            log.warning(f"No _target_ specified for optimizer, using default Adam")
            opt_cfg["_target_"] = "torch.optim.Adam"

        log.info(f"Instantiating optimizer <{opt_cfg._target_}>")
        optimizer = instantiate(opt_cfg, self.parameters())

        if "scheduler" not in train_cfg:
            return optimizer
        else:
            log.info(f"Instantiating scheduler <{train_cfg.scheduler._target_}>")
            lr_scheduler = instantiate(train_cfg.scheduler, optimizer)
            return [optimizer], {
                "scheduler": lr_scheduler,
                "interval": train_cfg.get("scheduler_interval", "step"),
                "monitor": train_cfg.get("scheduler_monitor", "val/loss"),
            }

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        out = self.model(td, phase, td.get("extra", None))

        # Log metrics
        metrics = getattr(self, f"{phase}_metrics")
        metrics = {f"{phase}/{k}": v.mean() for k, v in out.items() if k in metrics}

        log_on_step = self.log_on_step if phase == "train" else False
        on_epoch = False if phase == "train" else True
        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return {"loss": out.get("loss", None), **metrics}

    def training_step(self, batch: Any, batch_idx: int):
        # To use new data every epoch, we need to call reload_dataloaders_every_epoch=True in Trainer
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, self.train_batch_size)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, self.val_batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_batch_size)

    def on_train_epoch_end(self):
        if hasattr(self.model, "on_train_epoch_end"):
            self.model.on_train_epoch_end(self)
        train_dataset = self.env.dataset(self.cfg.data.train_size, "train")
        self.train_dataset = self.wrap_dataset(train_dataset)

    def wrap_dataset(self, dataset):
        if hasattr(self.model, "wrap_dataset"):
            dataset = self.model.wrap_dataset(self, dataset)
        return dataset

    def _dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # no need to shuffle, we're resampling every epoch
            num_workers=self.cfg.data.get("num_workers", 0),
            collate_fn=tensordict_collate_fn,
        )
