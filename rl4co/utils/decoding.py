from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import batchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def get_decoding_strategy(decoding_strategy, **config):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "multistart_greedy": Greedy,
        "multistart_sampling": Sampling,
        "beam_search": BeamSearch,
    }

    if decoding_strategy not in strategy_registry:
        log.warning(
            f"Unknown decode type '{decoding_strategy}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    if "multistart" in decoding_strategy:
        config["multistart"] = True

    return strategy_registry.get(decoding_strategy, Sampling)(**config)


def get_log_likelihood(logprobs, actions, mask=None, return_sum: bool = True):
    """Get log likelihood of selected actions.
    Note that mask is a boolean tensor where True means the value should be kept.

    Args:
        logprobs: Log probabilities of actions from the model (batch_size, seq_len, action_dim).
        actions: Selected actions (batch_size, seq_len).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        return_sum: Whether to return the sum of log probabilities or not. Defaults to True.
    """
    logprobs = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        logprobs[~mask] = 0
    # TODO: check

    assert (
        logprobs > -1000
    ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate log_likelihood
    if return_sum:
        return logprobs.sum(1)  # [batch]
    else:
        return logprobs  # [batch, decode_len]


def decode_logprobs(logprobs, mask, decode_type="sampling"):
    """Decode log probabilities to select actions with mask.
    Note that mask is a boolean tensor where True means the value should be kept.
    """
    if "greedy" in decode_type:
        selected = DecodingStrategy.greedy(logprobs, mask)
    elif "sampling" in decode_type:
        selected = DecodingStrategy.sampling(logprobs, mask)
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


def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done out-of-place.
    Ref: https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/sample.py#L6
    """
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, float("-inf"))


def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done out-of-place.
    Ref: https://github.com/togethercomputer/stripedhyena/blob/7e13f618027fea9625be1f2d2d94f9a361f6bd02/stripedhyena/sample.py#L14
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def process_logits(
    logits: torch.Tensor,
    mask: torch.Tensor = None,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    tanh_clipping: float = 0,
    mask_logits: bool = True,
):
    """Convert logits to log probabilities with additional features like temperature scaling, top-k and top-p sampling.

    Note:
        We convert to log probabilities instead of probabilities to avoid numerical instability.
        This is because, roughly, softmax = exp(logits) / sum(exp(logits)) and log(softmax) = logits - log(sum(exp(logits))),
        and avoiding the division by the sum of exponentials can help with numerical stability.
        You may check the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html).

    Args:
        logits: Logits from the model (batch_size, num_actions).
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation).
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Remove tokens that have a cumulative probability
            less than the threshold 1 - top_p (lower tail of the distribution). If 0, do not perform.
        top_k: Top-k sampling, i.e. restrict sampling to the top k logits. If 0, do not perform. Note that we only do filtering and
            do not return all the top-k logits here.
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940).
        mask_logits: Whether to mask logits of infeasible actions.
    """

    # Tanh clipping from Bello et al. 2016
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    # In RL, we want to mask the logits to prevent the agent from selecting infeasible actions
    if mask_logits:
        assert mask is not None, "mask must be provided if mask_logits is True"
        logits[~mask] = float("-inf")

    logits = logits / temperature  # temperature scaling

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # safety check
        logits = modify_logits_for_top_k_filtering(logits, top_k)

    if top_p > 0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
        logits = modify_logits_for_top_p_filtering(logits, top_p)

    # Compute log probabilities
    return F.log_softmax(logits, dim=-1)


class DecodingStrategy:
    """Base class for decoding strategies. Subclasses should implement the :meth:`_step` method.
    Includes hooks for pre and post main decoding operations.

    Args:
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation). Defaults to 1.0.
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Defaults to 0.0.
        top_k: Top-k sampling, i.e. restrict sampling to the top k logits. If 0, do not perform. Defaults to 0.
        mask_logits: Whether to mask logits of infeasible actions. Defaults to True.
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940). Defaults to 0.
        multistart: Whether to use multistart decoding. Defaults to False.
        num_starts: Number of starts for multistart decoding. Defaults to None.
    """

    name = "base"

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        mask_logits: bool = True,
        tanh_clipping: float = 0,
        multistart: bool = False,
        num_starts: Optional[int] = None,
        select_start_nodes_fn: Optional[callable] = None,
        **kwargs,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.multistart = multistart
        self.num_starts = num_starts
        self.select_start_nodes_fn = select_start_nodes_fn
        # initialize buffers
        self.actions = []
        self.logprobs = []

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Main decoding operation. This method should be called in a loop until all sequences are done.
        Args:
            logprobs: Log probabilities processed from logits of the model.
            mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
            td: TensorDict containing the current state of the environment.
        """
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(self, td: TensorDict, env: RL4COEnvBase):
        """Pre decoding hook. This method is called before the main decoding operation."""
        # Multi-start decoding. If num_starts is None, we use the number of actions in the action mask
        if self.multistart:
            if self.num_starts is None:
                self.num_starts = env.get_num_starts(td)
        else:
            if self.num_starts is not None:
                if self.num_starts >= 1:
                    log.warn(
                        f"num_starts={self.num_starts} is ignored for decode_type={self.name}"
                    )

            self.num_starts = 0

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.num_starts >= 1:
            if self.select_start_nodes_fn is not None:
                action = self.select_start_nodes_fn(td, env, self.num_starts)
            else:
                action = env.select_start_nodes(td, num_starts=self.num_starts)

            # Expand td to batch_size * num_starts
            td = batchify(td, self.num_starts)

            td.set("action", action)
            td = env.step(td)["next"]
            logprobs = torch.zeros_like(
                td["action_mask"], device=td.device
            )  # first logprobs is 0, so p = logprobs.exp() = 1

            self.logprobs.append(logprobs)
            self.actions.append(action)

        return td, env, self.num_starts

    def post_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
        assert (
            len(self.logprobs) > 0
        ), "No logprobs were collected because all environments were done. Check your initial state"

        return torch.stack(self.logprobs, 1), torch.stack(self.actions, 1), td, env

    def step(
        self, logits: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> TensorDict:
        """Main decoding operation. This method should be called in a loop until all sequences are done.
        Args:
            logits: Logits from the model.
            mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
            td: TensorDict containing the current state of the environment.
        """
        if not self.mask_logits:  # set mask_logit to None if mask_logits is False
            mask = None

        logprobs = process_logits(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
        )
        logprobs, selected_actions, td = self._step(logprobs, mask, td, **kwargs)
        td.set("action", selected_actions)
        self.actions.append(selected_actions)
        self.logprobs.append(logprobs)
        return td

    @staticmethod
    def greedy(logprobs, mask=None):
        """Select the action with the highest probability."""
        # [BS], [BS]
        selected = logprobs.argmax(dim=-1)
        if mask is not None:
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected

    @staticmethod
    def sampling(logprobs, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""
        probs = logprobs.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)

        if mask is not None:
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                log.info("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected


class Greedy(DecodingStrategy):
    name = "greedy"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select the action with the highest log probability."""
        selected = self.greedy(logprobs, mask)
        return logprobs, selected, td


class Sampling(DecodingStrategy):
    name = "sampling"

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Sample an action with a multinomial distribution given by the log probabilities."""
        selected = self.sampling(logprobs, mask)
        return logprobs, selected, td


class BeamSearch(DecodingStrategy):
    name = "beam_search"

    def __init__(self, beam_width=None, select_best=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beam_width = beam_width
        self.select_best = select_best
        self.parent_beam_logprobs = None
        self.beam_path = []

    def _step(
        self, logprobs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        selected, batch_beam_idx = self._make_beam_step(logprobs)
        # select the correct state representation, logprobs and mask according to beam parent
        td = td[batch_beam_idx]
        logprobs = logprobs[batch_beam_idx]
        mask = mask[batch_beam_idx]

        assert (
            not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
        ), "infeasible action selected"

        return logprobs, selected, td

    def pre_decoder_hook(self, td: TensorDict, env: RL4COEnvBase, **kwargs):
        if self.beam_width is None:
            self.beam_width = env.get_num_starts(td)
        assert self.beam_width > 1, "beam width must be larger than 1"

        # select start nodes. TODO: include first step in beam search as well
        if self.select_start_nodes_fn is not None:
            action = self.select_start_nodes_fn(td, env, self.beam_width)
        else:
            action = env.select_start_nodes(td, num_starts=self.beam_width)

        # Expand td to batch_size * beam_width
        td = batchify(td, self.beam_width)

        td.set("action", action)
        td = env.step(td)["next"]

        logprobs = torch.zeros_like(td["action_mask"], device=td.device)
        beam_parent = torch.zeros(logprobs.size(0), device=td.device, dtype=torch.int32)

        self.logprobs.append(logprobs)
        self.actions.append(action)
        self.parent_beam_logprobs = logprobs.gather(1, action[..., None])
        self.beam_path.append(beam_parent)

        return td, env, self.beam_width

    def post_decoder_hook(self, td, env):
        # [BS*BW, seq_len]
        aligned_sequences, aligned_logprobs = self._backtrack()

        if self.select_best:
            return self._select_best_beam(aligned_logprobs, aligned_sequences, td, env)
        else:
            return aligned_logprobs, aligned_sequences, td, env

    def _backtrack(self):
        # [BS*BW, seq_len]
        actions = torch.stack(self.actions, 1)
        # [BS*BW, seq_len]
        logprobs = torch.stack(self.logprobs, 1)
        assert actions.size(1) == len(
            self.beam_path
        ), "action idx shape and beam path shape dont match"

        # [BS*BW]
        cur_parent = self.beam_path[-1]
        # [BS*BW]
        reversed_aligned_sequences = [actions[:, -1]]
        reversed_aligned_logprobs = [logprobs[:, -1]]

        aug_batch_size = actions.size(0)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(actions.device)
        )

        for k in reversed(range(len(self.beam_path) - 1)):
            batch_beam_idx = batch_beam_sequence + cur_parent * batch_size

            reversed_aligned_sequences.append(actions[batch_beam_idx, k])
            reversed_aligned_logprobs.append(logprobs[batch_beam_idx, k])
            cur_parent = self.beam_path[k][batch_beam_idx]

        # [BS*BW, seq_len*num_targets]
        actions = torch.stack(list(reversed(reversed_aligned_sequences)), dim=1)
        logprobs = torch.stack(list(reversed(reversed_aligned_logprobs)), dim=1)

        return actions, logprobs

    def _select_best_beam(self, logprobs, actions, td: TensorDict, env: RL4COEnvBase):
        aug_batch_size = logprobs.size(0)  # num nodes
        batch_size = aug_batch_size // self.beam_width
        rewards = env.get_reward(td, actions)
        _, idx = torch.cat(rewards.unsqueeze(1).split(batch_size), 1).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return logprobs[flat_idx], actions[flat_idx], td[flat_idx], env

    def _make_beam_step(self, logprobs: torch.Tensor):
        aug_batch_size, num_nodes = logprobs.shape  # num nodes
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(logprobs.device)
        )

        # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
        log_beam_prob = logprobs + self.parent_beam_logprobs  #

        # [BS, num_nodes * BW]
        log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)
        # [BS, BW]
        topk_logprobs, topk_ind = torch.topk(
            log_beam_prob_hstacked, self.beam_width, dim=1
        )

        # [BS*BW, 1]
        logprobs_selected = torch.hstack(torch.unbind(topk_logprobs, 1)).unsqueeze(1)

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind, 1))

        # since we stack the logprobs from the distinct branches, the indices in
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index

        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        self.parent_beam_logprobs = logprobs_selected
        self.beam_path.append(beam_parent)

        return selected, batch_beam_idx
