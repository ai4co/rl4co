import lightning as L
import torch


def get_lightning_device(lit_module: L.LightningModule) -> torch.device:
    """Get the device of the Lightning module before setup is called
    See device setting issue in setup https://github.com/Lightning-AI/lightning/issues/2638
    """
    if lit_module.trainer.strategy.root_device != lit_module.device:
        return lit_module.trainer.strategy.root_device
    return lit_module.device
