from typing import Union
from tensordict import TensorDict
from torch import Tensor


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
