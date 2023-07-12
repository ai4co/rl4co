import torch
from torch.optim import Optimizer


def get_pytorch_lr_schedulers():
    return torch.optim.lr_scheduler.__all__


def create_scheduler(optimizer: Optimizer, scheduler_name: str, **scheduler_kwargs):
    if scheduler_name in get_pytorch_lr_schedulers():
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        return scheduler_cls(optimizer, **scheduler_kwargs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found.")
