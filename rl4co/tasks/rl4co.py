from typing import Any

import torch
import torch.nn as nn

from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rl4co.data.dataset import tensordict_collate_fn
from rl4co.data.generate_data import generate_default_datasets
from rl4co.envs.common.base import RL4COEnvBase
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

    def __init__(
        self, cfg: DictConfig, env: RL4COEnvBase = None, model: nn.Module = None
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # self.save_hyperparameters("env", "model", logger=False)
        self.save_hyperparameters(logger=False)

        if cfg.get("train", {}).get("disable_profiling", True):
            # Disable profiling executor. This reduces memory and increases speed.
            # https://github.com/HazyResearch/safari/blob/111d2726e7e2b8d57726b7a8b932ad8a4b2ad660/train.py#LL124-L129C17
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
            except AttributeError:
                pass

        cfg = DictConfig(cfg) if not isinstance(cfg, DictConfig) else cfg
        self.cfg = cfg

        # Instantiate environment, model and metrics
        self.env = env if env is not None else self.instantiate_env()
        self.model = model if model is not None else self.instantiate_model()
        self.instantiate_metrics()

        if cfg.get("train", {}).get("manual_optimization", False):
            log.info("Manual optimization enabled")
            self.automatic_optimization = False

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
            log.info("No metrics specified, using default")
        self.train_metrics = metrics.get("train", ["loss", "reward"])
        self.val_metrics = metrics.get("val", ["reward"])
        self.test_metrics = metrics.get("test", ["reward"])
        self.log_on_step = metrics.get("log_on_step", True)

    def setup(self, stage="fit"):
        log.info("Setting up batch sizes for train/val/test")
        # If any of the batch sizes are specified, use that. Otherwise, use the default batch size

        data_cfg = self.cfg.get("data", {})
        batch_size = data_cfg.get("batch_size", None)
        if data_cfg.get("train_batch_size", None) is not None:
            train_batch_size = data_cfg.train_batch_size
            if batch_size is not None:
                log.warning(
                    f"`train_batch_size`={train_batch_size} specified, ignoring `batch_size`={batch_size}"
                )
        elif batch_size is not None:
            train_batch_size = batch_size
        else:
            train_batch_size = 64
            log.warning(f"No batch size specified, using default as {train_batch_size}")
        # default all batch sizes to train_batch_size if not specified
        self.train_batch_size = train_batch_size
        self.val_batch_size = data_cfg.get("val_batch_size", train_batch_size)
        self.test_batch_size = data_cfg.get("test_batch_size", train_batch_size)

        log.info("Setting up datasets")

        # Create datasets automatically. If found, this will skip
        if data_cfg.get("generate_data", True):
            generate_default_datasets(
                data_dir=self.cfg.get("paths", {}).get("data_dir", "data/")
            )

        # If any of the dataset sizes are specified, use that. Otherwise, use the default dataset size
        def _get_phase_size(phase):
            DEFAULT_SIZES = {
                "train": 100000,
                "val": 10000,
                "test": 10000,
            }
            size = data_cfg.get(f"{phase}_size", None)
            if size is None:
                size = DEFAULT_SIZES[phase]
                message = f"No {phase}_size specified, using default as {size}"
                log.warning(message) if phase == "train" else log.info(message)
            return size

        self.train_size = _get_phase_size("train")
        self.val_size = _get_phase_size("val")
        self.test_size = _get_phase_size("test")
        self.train_dataset = self.wrap_dataset(self.env.dataset(self.train_size, "train"))
        self.val_dataset = self.env.dataset(self.val_size, "val")
        self.test_dataset = self.env.dataset(self.test_size, "test")

        if hasattr(self.model, "setup") and not self.cfg.get(
            "disable_model_setup", False
        ):
            self.model.setup(self)

    def configure_optimizers(self):
        train_cfg = self.cfg.get("train", {})
        if train_cfg.get("optimizer", None) is None:
            log.warning("No optimizer specified, using default")
        opt_cfg = train_cfg.get(
            "optimizer", DictConfig({"_target_": "torch.optim.Adam", "lr": 1e-4})
        )
        if "_target_" not in opt_cfg:
            log.info("No _target_ specified for optimizer, using default Adam")
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
                "interval": train_cfg.get("scheduler_interval", "epoch"),
                "monitor": train_cfg.get("scheduler_monitor", "val/reward"),
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
        train_dataset = self.env.dataset(self.train_size, "train")
        self.train_dataset = self.wrap_dataset(train_dataset)

    def wrap_dataset(self, dataset):
        if hasattr(self.model, "wrap_dataset") and not self.cfg.get(
            "disable_wrap_dataset", False
        ):
            dataset = self.model.wrap_dataset(self, dataset)
        return dataset

    def _dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # no need to shuffle, we're resampling every epoch
            num_workers=self.cfg.get("data", {}).get("num_workers", 0),
            collate_fn=tensordict_collate_fn,
        )
