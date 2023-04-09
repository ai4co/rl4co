"""Basic utilities for common tasks in Python and PyTorch."""
import re
from pathlib import Path

import torch


def flatten_params(params):
    """Flatten an iterable of parameters."""
    flat_params = [p.contiguous().view(-1) for p in params]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def flatten_params_grad(params, params_ref):
    """Flatten an iterable of parameters and their gradients."""
    _params = [p for p in params]
    _params_ref = [p for p in params_ref]
    flat_params = [
        p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(_params, _params_ref)
    ]
    return torch.cat(flat_params) if len(flat_params) > 0 else torch.tensor([])


def parameter_count(model):
    "Returns parameter count of an nn.Module."
    return sum([p.numel() for p in model.parameters()])


def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all(x >= y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    return all(x <= y for x, y in zip(L, L[1:]))


def monotonic(L):
    return non_increasing(L) or non_decreasing(L)


def find(tensor, values):
    "Finds indices of elements in a tensor that are equal to values."
    return torch.nonzero(tensor[..., None] == values)


def sum_except(x, num_dims=1):
    """
    Sums all dimensions except the first `num_dims`.
    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)
    Returns:
        x_sum: Tensor, shape (batch_size,)
    """
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def load_checkpoint(path, device="cpu"):
    "Loads nn.Module from a path."
    path = Path(path).expanduser()
    is_deepspeed = False
    if path.is_dir():  # DeepSpeed checkpoint
        is_deepspeed = True
        latest_path = path / "latest"
        if latest_path.is_file():
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        path /= f"{tag}/mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location=device)
    if is_deepspeed:
        state_dict = state_dict["module"]

        # Replace the names of some of the submodules
        def key_mapping(key):
            return re.sub(r"^module.model.", "", key)

        state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    return state_dict
