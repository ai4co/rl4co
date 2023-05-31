import torch

from rl4co.utils.ops import unbatchify


def select_start_nodes(batch_size, num_nodes, device="cpu", problem="tsp"):
    """Node selection strategy for POMO
    Selects different start nodes for each batch element
    """
    selected = torch.arange(num_nodes, device=device).repeat_interleave(batch_size)
    return selected


def get_best_actions(actions, max_idxs):
    actions = unbatchify(actions, max_idxs.shape[0])
    return actions.gather(0, max_idxs[..., None, None])
