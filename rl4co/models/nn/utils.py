from typing import Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils import get_pylogger

log = get_pylogger(__name__)


def get_log_likelihood(log_p, actions, mask, return_sum: bool = True):
    """Get log likelihood of selected actions"""

    log_p = log_p.gather(2, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[~mask] = 0

    assert (
        log_p > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return log_p.sum(1)  # [batch]
    else:
        return log_p  # [batch, decode_len]


def decode_probs(probs, mask, decode_type="sampling"):
    """Decode probabilities to select actions with mask"""

    assert (probs == probs).all(), "Probs should not contain any nans"

    if "greedy" in decode_type:
        _, selected = probs.max(1)
        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "Decode greedy: infeasible action has maximum probability"

    elif "sampling" in decode_type:
        selected = torch.multinomial(probs, 1).squeeze(1)

        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            log.info("Sampled bad values, resampling!")
            selected = probs.multinomial(1).squeeze(1)

    else:
        assert False, "Unknown decode type: {}".format(decode_type)
    return selected


def random_policy(td):
    """Helper function to select a random action from available actions"""
    action = torch.multinomial(td["action_mask"].float(), 1).squeeze(-1)
    td.set("action", action)
    return td


def rollout(env, td, policy, max_steps: int = None):
    """Helper function to rollout a policy. Currently, TorchRL does not allow to step
    over envs when done with `env.rollout()`. We need this because for environments that complete at different steps.
    """

    max_steps = float("inf") if max_steps is None else max_steps
    actions = []
    steps = 0

    while not td["done"].all():
        td = policy(td)
        actions.append(td["action"])
        td = env.step(td)["next"]
        steps += 1
        if steps > max_steps:
            log.info("Max steps reached")
            break
    return (
        env.get_reward(td, torch.stack(actions, dim=1)),
        td,
        torch.stack(actions, dim=1),
    )


class RandomPolicy(nn.Module):
    """
    Random Policy Class that randomly select actions from the action space
    This policy can be useful to check the sanity of the environment during development

    We match the function signature of forward to the one of the AutoregressivePolicy class

    """

    def __init__(self, env_name=None):
        super().__init__()
        self.env_name = env_name

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
        max_steps: int = None,
    ):
        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        return rollout(env, td, random_policy, max_steps=max_steps)
