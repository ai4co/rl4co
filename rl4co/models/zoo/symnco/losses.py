from torch.nn.functional import cosine_similarity

from rl4co.utils.ops import unbatchify


def problem_symmetricity_loss(reward, log_likelihood, dim=1):
    """REINFORCE loss for problem symmetricity
    Baseline is the average reward for all augmented problems
    Corresponds to `L_ps` in the SymNCO paper
    """
    num_augment = reward.shape[dim]
    if num_augment < 2:
        return 0
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -advantage * log_likelihood
    return loss.mean()


def solution_symmetricity_loss(reward, log_likelihood, dim=-1):
    """REINFORCE loss for solution symmetricity
    Baseline is the average reward for all start nodes
    Corresponds to `L_ss` in the SymNCO paper
    """
    num_starts = reward.shape[dim]
    if num_starts < 2:
        return 0
    advantage = reward - reward.mean(dim=dim, keepdim=True)
    loss = -advantage * log_likelihood
    return loss.mean()


def invariance_loss(proj_embed, num_augment):
    """Negative cosine similarity loss for invariant projected node representations.
    Corresponds to `L_inv` in the SymNCO paper
    """
    pe = unbatchify(proj_embed, num_augment)
    similarity = sum([cosine_similarity(pe[:, 0], pe[:, i], dim=-1) for i in range(1, num_augment)])
    return -similarity.mean()
