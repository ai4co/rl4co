import torch

from rl4co.utils import get_pylogger

log = get_pylogger(__name__)


def get_log_likelihood(log_p, actions, mask):
    """Get log likelihood of selected actions"""

    log_p = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[~mask] = 0

    assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    return log_p.sum(1)


def get_entropy(log_p, mask):
    """
    masked entropy of log_p
    """
    entropy = -log_p * log_p.exp().sum(dim=1)
    return entropy


def decode_probs(probs, mask, decode_type="sampling"):
    """Decode probabilities to select actions with mask"""

    assert (probs == probs).all(), "Probs should not contain any nans"

    if decode_type == "greedy":
        _, selected = probs.max(1)
        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "Decode greedy: infeasible action has maximum probability"

    elif decode_type == "sampling":
        selected = torch.multinomial(probs, 1).squeeze(1)

        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            log.info("Sampled bad values, resampling!")
            selected = probs.multinomial(1).squeeze(1)

    else:
        assert False, "Unknown decode type"
    return selected


def random_policy(td):
    """Helper function to select a random action from available actions"""
    action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
    td.set("action", action)
    return td