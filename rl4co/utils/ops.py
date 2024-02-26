from typing import Union

import torch

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor


def _batchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)


@torch.jit.script
def get_distance(x: Tensor, y: Tensor):
    """Euclidean distance between two tensors of shape `[..., n, dim]`"""
    return (x - y).norm(p=2, dim=-1)


@torch.jit.script
def get_tour_length(ordered_locs):
    """Compute the total tour distance for a batch of ordered tours.
    Computes the L2 norm between each pair of consecutive nodes in the tour and sums them up.

    Args:
        ordered_locs: Tensor of shape [batch_size, num_nodes, 2] containing the ordered locations of the tour
    """
    ordered_locs_next = torch.roll(ordered_locs, 1, dims=-2)
    return get_distance(ordered_locs_next, ordered_locs).sum(-1)


def get_num_starts(td, env_name=None):
    """Returns the number of possible start nodes for the environment based on the action mask"""
    num_starts = td["action_mask"].shape[-1]
    if env_name == "pdp":
        num_starts = (
            num_starts - 1
        ) // 2  # only half of the nodes (i.e. pickup nodes) can be start nodes
    elif env_name in ["cvrp", "sdvrp", "mtsp", "op", "pctsp", "spctsp"]:
        num_starts = num_starts - 1  # depot cannot be a start node
    return num_starts


def select_start_nodes(td, env, num_starts):
    """Node selection strategy as proposed in POMO (Kwon et al. 2020)
    and extended in SymNCO (Kim et al. 2022).
    Selects different start nodes for each batch element

    Args:
        td: TensorDict containing the data. We may need to access the available actions to select the start nodes
        env: Environment may determine the node selection strategy
        num_starts: Number of nodes to select. This may be passed when calling the policy directly. See :class:`rl4co.models.AutoregressiveDecoder`
    """
    if env.name in ["tsp", "atsp"]:
        selected = torch.arange(num_starts, device=td.device).repeat_interleave(
            td.shape[0]
        )
    else:
        # Environments with depot: we do not select the depot as a start node
        selected = torch.arange(1, num_starts + 1, device=td.device).repeat_interleave(
            td.shape[0]
        )
        if env.name == "op":
            if (td["action_mask"][..., 1:].float().sum(-1) < num_starts).any():
                # for the orienteering problem, we may have some nodes that are not available
                # so we need to resample from the distribution of available nodes
                selected = (
                    torch.multinomial(
                        td["action_mask"][..., 1:].float(), num_starts, replacement=True
                    )
                    + 1
                )  # re-add depot index
                selected = rearrange(selected, "b n -> (n b)")
    return selected


def get_best_actions(actions, max_idxs):
    actions = unbatchify(actions, max_idxs.shape[0])
    return actions.gather(0, max_idxs[..., None, None])
