import inspect

import torch
import torch.nn as nn


def get_pytorch_optimizers():
    optimizers = []
    for name, obj in inspect.getmembers(torch.optim):
        if inspect.isclass(obj) and issubclass(obj, torch.optim.Optimizer):
            optimizers.append(name)
    return optimizers


def create_optimizer(model: nn.Module, optimizer_name: str, **optimizer_kwargs):
    if optimizer_name in get_pytorch_optimizers():
        optimizer_cls = getattr(torch.optim, optimizer_name)
        return optimizer_cls(model.parameters(), **optimizer_kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not found.")
