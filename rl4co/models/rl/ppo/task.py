from typing import Any

import torch.nn as nn

from omegaconf import DictConfig

from rl4co.envs.base import EnvBase
from rl4co.tasks.rl4co import RL4COLitModule


class PPOTask(RL4COLitModule):
    def __init__(self, cfg: DictConfig, env: EnvBase = None, model: nn.Module = None):
        super().__init__(cfg=cfg, env=env, model=model)
        self.automatic_optimization = False

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        out = self.model(
            td,
            phase,
            td.get("extra", None),
            optimizer=self.optimizers() if phase == "train" else None,
        )

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
