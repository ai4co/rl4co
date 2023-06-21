import os

import lightning as L
import torch
import yaml

from omegaconf import DictConfig

from rl4co.tasks.rl4co import RL4COLitModule
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_lightning_device(lit_module: L.LightningModule) -> torch.device:
    """Get the device of the Lightning module before setup is called
    See device setting issue in setup https://github.com/Lightning-AI/lightning/issues/2638
    """
    if lit_module.trainer.strategy.root_device != lit_module.device:
        return lit_module.trainer.strategy.root_device
    return lit_module.device


def remove_key(config, key="wandb"):
    """Remove keys containing 'key`"""
    new_config = {}
    for k, v in config.items():
        if key in k:
            continue
        else:
            new_config[k] = v
    return new_config


def clean_hydra_config(
    config, keep_value_only=True, remove_keys="wandb", clean_cfg_path=True
):
    """Clean hydra config by nesting dictionary and cleaning values"""
    # Remove keys containing `remove_keys`
    if not isinstance(remove_keys, list):
        remove_keys = [remove_keys]
    for key in remove_keys:
        config = remove_key(config, key=key)

    new_config = {}
    # Iterate over config dictionary
    for key, value in config.items():
        # If key contains slash, split it and create nested dictionary recursively
        if "/" in key:
            keys = key.split("/")
            d = new_config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value["value"] if keep_value_only else value
        else:
            new_config[key] = value["value"] if keep_value_only else value

    cfg = DictConfig(new_config)

    if clean_cfg_path:
        # Clean cfg_path recursively substituting root_dir with cwd
        root_dir = cfg.paths.root_dir

        def replace_dir_recursive(d, search, replace):
            for k, v in d.items():
                if isinstance(v, dict) or isinstance(v, DictConfig):
                    replace_dir_recursive(v, search, replace)
                elif isinstance(v, str):
                    if search in v:
                        d[k] = v.replace(search, replace)

        replace_dir_recursive(cfg, root_dir, os.getcwd())

    return cfg


def load_model_from_checkpoint(
    config,
    checkpoint_path,
    device="cpu",
    only_policy=True,
    disable_model_setup=True,
    disable_wrap_dataset=True,
    validate_only=True,
    clean_cfg_path=True,
    phase="test",
):
    """Load model from checkpoint

    Args:
        config: Hydra config or its path
        checkpoint_path: Path to checkpoint
        device: Device to load model on
        only_policy: If True, load only policy parameters
        disable_model_setup: If True, disable model setup during RL4COLitModule init
        disable_wrap_dataset: If True, disable dataset wrapping during RL4COLitModule init
        validate_only: If True, only load model for validation and make train size small
    """
    if only_policy and not (disable_model_setup or disable_wrap_dataset):
        log.warning(
            "only_policy is True, but disable_model_setup and disable_wrap_dataset are False. "
            "This may cause errors due to missing model setup and dataset wrapping. "
        )

    # Load config if path is given
    if not isinstance(config, DictConfig or dict):
        log.info(f"Loading config from {config}")
        with open(config, "r") as stream:
            config = yaml.safe_load(stream)

    # Clean hydra config
    config = clean_hydra_config(config, clean_cfg_path=clean_cfg_path)

    # Add to cfg disable_model_setup and disable_wrap_dataset
    config["disable_model_setup"] = disable_model_setup
    config["disable_wrap_dataset"] = disable_wrap_dataset
    if validate_only:
        config["train_size"] = 10  # dummy

    # Load model and checkpoint
    lit_module = RL4COLitModule(config)
    checkpoint_path = torch.load(checkpoint_path, map_location=device)

    # Load model from checkpoint: only policy parameters or full model
    if only_policy:
        state_dict = checkpoint_path["state_dict"]
        # get only policy parameters
        state_dict = {k: v for k, v in state_dict.items() if "policy" in k}
        # remove leading 'policy.' from keys
        state_dict = {k.replace("model.policy.", ""): v for k, v in state_dict.items()}
        # load policy state_dict
        lit_module.model.policy.load_state_dict(state_dict)
    else:
        lit_module = lit_module.load_from_checkpoint(checkpoint_path)

    lit_module.setup(stage=phase)
    return lit_module
