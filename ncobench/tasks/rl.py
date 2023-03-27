from typing import List, Tuple, Optional, NamedTuple, Dict, Union, Any
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader
import lightning as L

from ncobench.data.dataset import TorchDictDataset
from ncobench.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class NCOLightningModule(L.LightningModule):
    def __init__(self, cfg, model_cfg=None, env_cfg=None):
        """
        Base LightningModule for Neural Combinatorial Optimization
        If model_cfg is passed, it will take precedence over cfg.model
        Likewise for env_cfg

        Args:
            cfg: OmegaConf config
            model_cfg: OmegaConf config for model
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model_cfg = model_cfg or self.cfg.model
        self.env_cfg = env_cfg or self.cfg.env

        self.instantiate_env()
        self.instantiate_model()
        self.setup()

    def instantiate_env(self):
        log.info(f"Instantiating environments <{self.env_cfg._target_}>")
        self.env = instantiate(self.env_cfg)

    def instantiate_model(self):
        log.info(f"Instantiating model <{self.model_cfg._target_}>")
        self.model = instantiate(self.model_cfg, env=self.env)

    def setup(self, stage="fit"):
        log.info(f"Setting up datasets")
        self.train_dataset = self.get_observation_dataset(self.data.train_size)
        self.val_dataset = self.get_observation_dataset(self.data.val_size)
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
        td = self.env.reset(init_observation=batch)
        output = self.model(td, phase)
        self.log(
            f"{phase}/cost",
            output["cost"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
        self.train_dataset = self.get_observation_dataset(self.train_size)

    def get_observation_dataset(self, size):
        # Online data generation: we generate a new batch online
        data = self.env.gen_params(batch_size=size)
        return TorchDictDataset(self.env.reset(data)["observation"])

    def _dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,  # no need to shuffle, we're resampling every epoch
            num_workers=self.cfg.data.get("num_workers", 0),
            collate_fn=torch.stack,  # we need this to stack the batches in the dataset
        )
