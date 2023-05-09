import sys

sys.path.append("./")

import math
from typing import List, Tuple, Optional, NamedTuple, Dict, Union, Any
from einops import rearrange, repeat
from hydra.utils import instantiate

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L

from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
from tensordict import TensorDict

from rl4co.data.dataset import TensorDictCollate, TensorDictDataset
from rl4co.models.rl.reinforce import *
from rl4co.models.nn.env_context import env_context
from rl4co.models.nn.env_embedding import env_init_embedding, env_dynamic_embedding
from rl4co.models.zoo.am.decoder import (
    Decoder,
    decode_probs,
    PrecomputedCache,
    LogitAttention,
)
from rl4co.models.zoo.am.policy import get_log_likelihood
from rl4co.models.zoo.am import AttentionModel, AttentionModelPolicy
from rl4co.models.nn.attention import NativeFlashMHA, flash_attn_wrapper
from rl4co.utils.lightning import get_lightning_device

from rl4co.envs.op import OPEnv


num_loc = 20
device = "cuda" if torch.cuda.is_available() else "cpu"
env = OPEnv(
    num_loc=num_loc,
    min_loc=0,
    max_loc=1,
    min_prize=1,
    max_prize=10,
    length_capacity=5,
    batch_size=[32],
    device=device,
)

# env = TSPEnv(num_loc=15).transform()
dataset = env.dataset(batch_size=[1000])

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,  # no need to shuffle, we're resampling every epoch
    num_workers=0,
    collate_fn=TensorDictCollate(),
)

policy = AttentionModelPolicy(
    env,
    embedding_dim=128,
    hidden_dim=128,
    num_encode_layers=3,
).to(device)

# NOTE: here x is a tensor dict instead of a tensor in tsp test
x = next(iter(dataloader)).to(device)
td = env.reset(tensordict=x)

out = policy(td, decode_type="sampling", return_actions=True)
print(out)


class NCOLightningModule(L.LightningModule):
    def __init__(
        self, env, model, lr=1e-4, batch_size=128, train_size=1000, val_size=10000
    ):
        super().__init__()

        # TODO: hydra instantiation
        self.env = env
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size

    def setup(self, stage="fit"):
        self.train_dataset = self.get_observation_dataset(self.train_size)
        self.val_dataset = self.get_observation_dataset(self.val_size)
        if hasattr(self.model, "setup"):
            self.model.setup(self)

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(tensordict=batch)
        output = self.model(td, phase)
        self.log(f"{phase}/cost", -output["reward"].mean(), prog_bar=True)
        return {"loss": output["loss"] if phase == "train" else None}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # optim = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
        # TODO: scheduler
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_steps)
        return [optim]  # , [scheduler]

    def train_dataloader(self):
        return self._dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def on_train_epoch_end(self):
        if hasattr(self.model, "on_train_epoch_end"):
            self.model.on_train_epoch_end(self)
        self.train_dataset = self.get_observation_dataset(self.train_size)

    def get_observation_dataset(self, size):
        # online data generation: we generate a new batch online
        # data = self.env.gen_params(batch_size=size)
        # FIXME: batch_size is required for the reset() function
        # return TensorDictDataset(self.env.reset(batch_size=[size])['observation'])
        return self.env.dataset(batch_size=[size])

    def _dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # no need to shuffle, we're resampling every epoch
            num_workers=0,
            collate_fn=TensorDictCollate(),
        )


batch_size = 32  # 1024 #512
epochs = 1
lr = 1e-4
train_size = 128000


# Instantiate full model
model = AttentionModel(env, policy)

# Lightning module
litmodel = NCOLightningModule(
    env, model, batch_size=batch_size, train_size=train_size, lr=lr
)

# Trick to make calculations faster
torch.set_float32_matmul_precision("medium")


# Trainer
trainer = L.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    log_every_n_steps=100,
    gradient_clip_val=1.0,  # clip gradients to avoid exploding gradients
)

# Fit the model
trainer.fit(litmodel)
