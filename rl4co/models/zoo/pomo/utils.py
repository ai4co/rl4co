import torch

from rl4co.utils.ops import undo_repeat_batch


def select_start_nodes(batch_size, num_nodes, device="cpu"):
    """
    Node selection strategy for POMO
    Selects different start nodes for each batch element, i.e. each different node
    """
    selected = torch.arange(num_nodes, device=device)
    return selected.repeat_interleave(batch_size, dim=0)


def get_best_actions(actions, max_idxs):
    """Get best actions from batches given max_idxs"""
    actions = undo_repeat_batch(actions, max_idxs.shape[0])
    return actions.gather(0, max_idxs[..., None, None])
