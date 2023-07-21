from typing import IO, Any, Optional, Union, cast

import torch
import torch.nn as nn

from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning.pytorch.core.saving import _load_from_checkpoint
from typing_extensions import Self

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline, get_reinforce_baseline
from rl4co.utils.lightning import get_lightning_device
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class REINFORCE(RL4COLitModule):
    """REINFORCE algorithm, also known as policy gradients.
    See superclass `RL4COLitModule` for more details.

    Args:
        env: Environment to use for the algorithm
        policy: Policy to use for the algorithm
        baseline: REINFORCE baseline
        baseline_kwargs: Keyword arguments for baseline. Ignored if baseline is not a string
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        baseline: Union[REINFORCEBaseline, str] = "rollout",
        baseline_kwargs={},
        **kwargs,
    ):
        super().__init__(env, policy, **kwargs)

        self.save_hyperparameters(logger=False)

        if isinstance(baseline, str):
            baseline = get_reinforce_baseline(baseline, **baseline_kwargs)
        else:
            if baseline_kwargs != {}:
                log.warning("baseline_kwargs is ignored when baseline is not a string")
        self.baseline = baseline

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        td = self.env.reset(batch)
        # Perform forward pass (i.e., constructing solution and computing log-likelihoods)
        out = self.policy(td, self.env, phase=phase)

        # Compute loss
        if phase == "train":
            # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
            extra = td.get("extra", None)

            bl_val, bl_neg_loss = (
                self.baseline.eval(td, out["reward"], self.env)
                if extra is None
                else (extra, 0)
            )

            advantage = out["reward"] - bl_val  # advantage = reward - baseline
            reinforce_loss = -(advantage * out["log_likelihood"]).mean()
            loss = reinforce_loss - bl_neg_loss
            out.update(
                {
                    "loss": loss,
                    "reinforce_loss": reinforce_loss,
                    "bl_loss": -bl_neg_loss,
                    "bl_val": bl_val,
                }
            )

        metrics = self.log_metrics(out, phase)
        return {"loss": out.get("loss", None), **metrics}

    def post_setup_hook(self, stage="fit"):
        # Make baseline taking model itself and train_dataloader from model as input
        self.baseline.setup(
            self.policy,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            dataset_size=self.data_cfg["val_data_size"],
        )

    def on_train_epoch_end(self):
        """Callback for end of training epoch: we evaluate the baseline"""
        self.baseline.epoch_callback(
            self.policy,
            env=self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
            epoch=self.current_epoch,
            dataset_size=self.data_cfg["val_data_size"],
        )
        # Need to call super() for the dataset to be reset
        super().on_train_epoch_end()

    def wrap_dataset(self, dataset):
        """Wrap dataset from baseline evaluation. Used in greedy rollout baseline"""
        return self.baseline.wrap_dataset(
            dataset,
            self.env,
            batch_size=self.val_batch_size,
            device=get_lightning_device(self),
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = False,
        load_baseline: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Load model from checkpoint/

        Note:
            This is a modified version of `load_from_checkpoint` from `pytorch_lightning.core.saving`.
            It deals with matching keys for the baseline by first running setup
        """

        if strict:
            log.warning("Setting strict=False for loading model from checkpoint.")
            strict = False

        # Do not use strict
        loaded = _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )

        # Load baseline state dict
        if load_baseline:
            # setup baseline first
            loaded.setup()
            loaded.post_setup_hook()
            # load baseline state dict
            state_dict = torch.load(checkpoint_path)["state_dict"]
            # get only baseline parameters
            state_dict = {k: v for k, v in state_dict.items() if "baseline" in k}
            state_dict = {k.replace("baseline.", "", 1): v for k, v in state_dict.items()}
            loaded.baseline.load_state_dict(state_dict)

        return cast(Self, loaded)
