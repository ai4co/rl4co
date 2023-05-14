from typing import Union

from tensordict import TensorDict
import torch
from torch import Tensor

# @torch.jit.script
def batchify(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well
    Same as einops.repeat(x, 'b ... -> (b r) ...', r=repeats) but ~1.5x faster
    """
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def unbatchify(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """Undoes repeat_batch operation for Tensordicts as well
    Same as einops.rearrange(x, '(r b) ... -> b r ...', r=repeats) but ~2x faster
    """
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


# @torch.jit.script
def gather_by_index(src, idx, dim=1):
    """Gather elements from src by index idx along specified dim
    For example:
        src: shape (64, 20, 2)
        idx: shape (64,)
        dim: 1  
    Returns:
        target: shape (64, 1, 2) 
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze()


def distance(x, y):
    """Euclidean distance between two tensors of shape (..., n, dim)"""
    return torch.norm(x - y, p=2, dim=-1)



