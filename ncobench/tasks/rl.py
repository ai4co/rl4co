from typing import List, Tuple, Optional, NamedTuple, Dict, Union, Any
from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from lightning import LightningModule

from ncobench.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class RL4COLitModule(LightningModule):
    def __init__(self, cfg, model_cfg=None, env_cfg=None):
        """
        Base LightningModule for Neural Combinatorial Optimization
        If model_cfg is passed, it will take precedence over cfg.model
        Likewise for env_cfg

        Args:
            cfg: OmegaConf config
            model_cfg: OmegaConf config for model
            env_cfg: OmegaConf config for env
        """

        if cfg.train.get("disable_profiling", True):
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
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model_cfg = model_cfg or self.cfg.model
        self.env_cfg = env_cfg or self.cfg.env

        self.instantiate_env()
        self.instantiate_model()

    def instantiate_env(self):
        log.info(f"Instantiating environments <{self.env_cfg._target_}>")
        env = instantiate(self.env_cfg)
        self.env = env.transform()  # transform to get the observations directly

    def instantiate_model(self):
        log.info(f"Instantiating model <{self.model_cfg._target_}>")
        self.model = instantiate(self.model_cfg, env=self.env)

    def setup(self, stage="fit"):
        log.info(f"Setting up datasets")
        self.train_dataset = self.env.dataset(self.cfg.data.train_size)
        self.val_dataset = self.env.dataset(self.cfg.data.val_size)
        if hasattr(self.model, "setup"):
            self.model.setup(self)

    def configure_optimizers(self):
        parameters = (
            self.parameters()
        )  # this will train task specific parameters such as Retrieval head for AAN
        log.info(f"Instantiating optimizer <{self.cfg.train.optimizer._target_}>")
        optimizer = instantiate(self.cfg.train.optimizer, parameters)

        if "scheduler" not in self.cfg.train:
            return optimizer
        else:
            log.info(f"Instantiating scheduler <{self.cfg.train.scheduler._target_}>")
            lr_scheduler = instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {
                "scheduler": lr_scheduler,
                "interval": self.cfg.train.get("scheduler_interval", "step"),
                "monitor": self.cfg.train.get("scheduler_monitor", "val/loss"),
            }

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(init_obs=batch)
        output = self.model(td, phase)

        # Choose whether to log reward or cost (negative reward)
        if self.cfg.train.get("log_cost", True):
            logged_metrics = {f"{phase}/cost": -output["reward"].mean()}
        else:
            logged_metrics = {f"{phase}/reward": output["reward"].mean()}
        logged_metrics[f"{phase}/loss"] = output["loss"].mean()

        self.log_dict(
            logged_metrics,
            on_step=self.cfg.train.get("log_on_step", True),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": output["loss"]}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def on_train_epoch_end(self):
        if hasattr(self.model, "on_train_epoch_end"):
            self.model.on_train_epoch_end(self)
        self.train_dataset = self.env.dataset(self.cfg.data.train_size)

    def _dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,  # no need to shuffle, we're resampling every epoch
            num_workers=self.cfg.data.get("num_workers", 0),
            collate_fn=torch.stack,  # we need this to stack the batches in the dataset
            # pin_memory=self.on_gpu, # TODO: check if needed, comment now for bug in test
        )


if __name__ == "__main__":
    from omegaconf import DictConfig
    from lightning import Trainer

    config = DictConfig(
        {
            "env": {
                "_target_": "ncobench.envs.tsp.TSPEnv",
                "num_loc": 50,
            },
            "model": {
                "_target_": "ncobench.models.am.AttentionModel",
                "policy": {
                    "_target_": "ncobench.models.components.am.base.AttentionModelBase",
                    "env": "${env}",
                },
                "baseline": {
                    "_target_": "ncobench.models.rl.reinforce.WarmupBaseline",
                    "baseline": {
                        "_target_": "ncobench.models.rl.reinforce.RolloutBaseline",
                    },
                },
            },
            "data": {
                "train_size": 1280000,
                "val_size": 10000,
                "batch_size": 512,
            },
            "train": {
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    # "_target_": "deepspeed.ops.adam.DeepSpeedCPUAdam",
                    "lr": 1e-4,
                    "weight_decay": 1e-5,
                },
                "gradient_clip_val": 1.0,
                "max_epochs": 100,
                "accelerator": "gpu",
            },
        }
    )

    torch.set_float32_matmul_precision("medium")

    # DDPS Strategy
    from lightning.pytorch.strategies import (
        DDPStrategy,
        FSDPStrategy,
        DeepSpeedStrategy,
    )

    model = NCOLitModule(config)
    trainer = Trainer(
        max_epochs=config.train.max_epochs,
        gradient_clip_val=config.train.gradient_clip_val,
        accelerator=config.train.accelerator,
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=[1],
        # strategy="deepspeed_stage_3",#(find_unused_parameters=True),
        # precision="16-mixed",
        precision=16,
        # accelerator="cpu",
    )
    trainer.fit(model)
