from typing import Tuple

import torch

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
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def logits_to_probs(
    logits, mask, temperature=1.0, top_p=0.0, tanh_clipping=0, mask_logits=True
):
    """Convert logits to probabilities. Optionally mask logits and apply temperature scaling.

    Args:
        logits: Logits from the model.
        mask: Action mask. 1 if feasible, 0 otherwise (so we keep if 1 as done in PyTorch).
        temperature: Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation).
        top_p: Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751).
        tanh_clipping: Tanh clipping (https://arxiv.org/abs/1611.09940).
        mask_logits: Whether to mask logits of infeasible actions.
    """

    # Tanh clipping from Bello et al. 2016
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping

    # In RL, we want to mask the logits to prevent the agent from selecting infeasible actions
    if mask_logits:
        logits[~mask] = float("-inf")

    logits = logits / temperature  # temperature scaling

    if top_p > 0:
        assert top_p <= 1.0, "top-p should be in (0, 1]."
        logits = modify_logits_for_top_p_filtering(logits, top_p)

    # Compute probabilities
    return torch.softmax(logits, dim=-1)


class DecodingStrategy:
    """Base class for decoding strategies. Subclasses should implement the :meth:`_step` method.
    Includes hooks for pre and post main decoding operations.

    Args:
        temperature (float, optional): Temperature scaling. Higher values make the distribution more uniform (exploration),
            lower values make it more peaky (exploitation). Defaults to 1.0.
        top_p (float, optional): Top-p sampling, a.k.a. Nucleus Sampling (https://arxiv.org/abs/1904.09751). Defaults to 0.0.
        mask_logits (bool, optional): Whether to mask logits of infeasible actions. Defaults to True.
        tanh_clipping (float, optional): Tanh clipping (https://arxiv.org/abs/1611.09940). Defaults to 0.
        multistart (bool, optional): Whether to use multistart decoding. Defaults to False.
        num_starts (int, optional): Number of starts for multistart decoding. Defaults to None.
    """

    name = "base"

    def __init__(
        self,
        temperature=1.0,
        top_p=0.0,
        mask_logits=True,
        tanh_clipping=0,
        multistart=False,
        num_starts=None,
        select_start_nodes_fn: callable = None,
        **kwargs,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.multistart = multistart
        self.num_starts = num_starts
        self.select_start_nodes_fn = select_start_nodes_fn
        # initialize buffers
        self.actions = []
        self.probs = []

    def _step(
        self, probs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Main decoding operation. This method should be called in a loop until all sequences are done.

        Args:
            probs: Probabilities from the model.
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
                if self.num_starts > 1:
                    log.warn(
                        f"num_starts={self.num_starts} is ignored for decode_type={self.name}"
                    )

            self.num_starts = 0

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.num_starts > 1:
            if self.select_start_nodes_fn is not None:
                action = self.select_start_nodes_fn(td, env, self.num_starts)
            else:
                action = env.select_start_nodes(td, num_starts=self.num_starts)

            # Expand td to batch_size * num_starts
            td = batchify(td, self.num_starts)

            td.set("action", action)
            td = env.step(td)["next"]
            prob = torch.ones_like(
                td["action_mask"], device=td.device
            )  # prob is 1 for the first action

            self.probs.append(prob)
            self.actions.append(action)

        return td, env, self.num_starts

    def post_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict, RL4COEnvBase]:
        """Returns the log probabilities, actions, TensorDict and environment after the main decoding operation."""
        assert (
            len(self.probs) > 0
        ), "No outputs were collected because all environments were done. Check your initial state"

        return torch.stack(self.probs, 1).log(), torch.stack(self.actions, 1), td, env

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
        probs = logits_to_probs(
            logits,
            mask,
            temperature=self.temperature,
            top_p=self.top_p,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
        )
        probs, selected_actions, td = self._step(probs, mask, td, **kwargs)
        td.set("action", selected_actions)
        self.actions.append(selected_actions)
        self.probs.append(probs)
        return td

    @staticmethod
    def greedy(probs, mask=None):
        """Select the action with the highest probability."""
        # [BS], [BS]
        selected = probs.argmax(dim=-1)
        if mask is not None:
            assert (
                not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
            ), "infeasible action selected"

        return selected

    @staticmethod
    def sampling(probs, mask=None):
        """Sample an action with a multinomial distribution given by the log probabilities."""
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _step(
        self, probs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        selected = self.greedy(probs, mask)
        return probs, selected, td


class Sampling(DecodingStrategy):
    name = "sampling"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _step(
        self, probs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        selected = self.sampling(probs, mask)
        return probs, selected, td


class BeamSearch(DecodingStrategy):
    name = "beam_search"

    def __init__(self, beam_width=None, select_best=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beam_width = beam_width
        self.select_best = select_best
        self.parent_beam_prob = None
        self.beam_path = []

    def _step(
        self, probs: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        selected, batch_beam_idx = self._make_beam_step(probs)
        # select the correct state representation, prob and mask according to beam parent
        td = td[batch_beam_idx]
        probs = probs[batch_beam_idx]
        mask = mask[batch_beam_idx]

        assert (
            not (~mask).gather(1, selected.unsqueeze(-1)).data.any()
        ), "infeasible action selected"

        return probs, selected, td

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

        probs = torch.ones_like(td["action_mask"], device=td.device)
        beam_parent = torch.ones(probs.size(0), device=td.device, dtype=torch.int32)

        self.probs.append(probs)
        self.actions.append(action)
        self.parent_beam_probs = probs.gather(1, action[..., None])
        self.beam_path.append(beam_parent)

        return td, env, self.beam_width

    def post_decoder_hook(self, td, env):
        # [BS*BW, seq_len]
        aligned_sequences, aligned_probs = self._backtrack()

        if self.select_best:
            return self._select_best_beam(aligned_probs, aligned_sequences, td, env)
        else:
            return aligned_probs, aligned_sequences, td, env

    def _backtrack(self):
        # [BS*BW, seq_len]
        actions = torch.stack(self.actions, 1)
        # [BS*BW, seq_len]
        probs = torch.stack(self.probs, 1)
        assert actions.size(1) == len(
            self.beam_path
        ), "action idx shape and beam path shape dont match"

        # [BS*BW]
        cur_parent = self.beam_path[-1]
        # [BS*BW]
        reversed_aligned_sequences = [actions[:, -1]]
        reversed_aligned_probs = [probs[:, -1]]

        aug_batch_size = actions.size(0)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(actions.device)
        )

        for k in reversed(range(len(self.beam_path) - 1)):
            batch_beam_idx = batch_beam_sequence + cur_parent * batch_size

            reversed_aligned_sequences.append(actions[batch_beam_idx, k])
            reversed_aligned_probs.append(probs[batch_beam_idx, k])
            cur_parent = self.beam_path[k][batch_beam_idx]

        # [BS*BW, seq_len*num_targets]
        actions = torch.stack(list(reversed(reversed_aligned_sequences)), dim=1)
        probs = torch.stack(list(reversed(reversed_aligned_probs)), dim=1)

        return actions, probs

    def _select_best_beam(self, probs, actions, td: TensorDict, env: RL4COEnvBase):
        aug_batch_size = probs.size(0)  # num nodes
        batch_size = aug_batch_size // self.beam_width
        rewards = env.get_reward(td, actions)
        _, idx = torch.cat(rewards.unsqueeze(1).split(batch_size), 1).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return probs[flat_idx], actions[flat_idx], td[flat_idx], env

    def _make_beam_step(self, probs: torch.Tensor):
        aug_batch_size, num_nodes = probs.shape  # num nodes
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(probs.device)
        )

        # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
        log_beam_prob = probs + self.parent_beam_probs  #

        # [BS, num_nodes * BW]
        log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)
        # [BS, BW]
        topk_probs, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1)

        # [BS*BW, 1]
        probs_selected = torch.hstack(torch.unbind(topk_probs, 1)).unsqueeze(1)

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind, 1))

        # since we stack the prob from the distinct branches, the indices in
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index

        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        self.parent_beam_probs = probs_selected
        self.beam_path.append(beam_parent)

        return selected, batch_beam_idx
