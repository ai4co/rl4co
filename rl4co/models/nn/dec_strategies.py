from typing import Tuple

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import batchify, get_num_starts, select_start_nodes
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


class DecodingStrategy:
    """Base class for decoding strategies. Subclasses should implement the :meth:`_step` method.
    Includes hooks for pre and post main decoding operations.

    Args:
        multistart (bool, optional): Whether to use multistart decoding. Defaults to False.
        num_starts (int, optional): Number of starts for multistart decoding. Defaults to None.
        select_start_nodes_fn (Callable, optional): Function to select start nodes. Defaults to select_start_nodes.
    """

    name = "base"

    def __init__(
        self,
        multistart=False,
        num_starts=None,
        select_start_nodes_fn=select_start_nodes,
        **kwargs,
    ) -> None:

        self.actions = []
        self.logp = []
        self.multistart = multistart
        self.num_starts = num_starts
        self.select_start_nodes_fn = select_start_nodes_fn

    def _step(
        self, logp: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main decoding operation. This method should be implemented by subclasses."""
        raise NotImplementedError("Must be implemented by subclass")

    def pre_decoder_hook(self, td: TensorDict, env: RL4COEnvBase):
        """Pre decoding hook. This method is called before the main decoding operation."""
        # Multi-start decoding. If num_starts is None, we use the number of actions in the action mask
        if self.multistart:
            if self.num_starts is None:
                self.num_starts = get_num_starts(td, env.name)
        else:
            if self.num_starts is not None:
                if self.num_starts > 1:
                    log.warn(
                        f"num_starts={self.num_starts} is ignored for decode_type={self.name}"
                    )

            self.num_starts = 0

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.num_starts > 1:
            action = self.select_start_nodes_fn(td, env, self.num_starts)

            # Expand td to batch_size * num_starts
            td = batchify(td, self.num_starts)

            td.set("action", action)
            td = env.step(td)["next"]
            log_p = torch.zeros_like(
                td["action_mask"], device=td.device
            )  # first log_p is 0, so p = log_p.exp() = 1

            self.logp.append(log_p)
            self.actions.append(action)

        return td, env, self.num_starts

    def post_decoder_hook(self, td, env):
        """Post decoding hook. This method is called after the main decoding operation."""
        assert (
            len(self.logp) > 0
        ), "No outputs were collected because all environments were done. Check your initial state"

        return torch.stack(self.logp, 1), torch.stack(self.actions, 1), td, env

    def step(
        self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Main decoding operation. This method calls the :meth:`_step` method and collects the outputs."""
        assert not logp.isinf().all(1).any()

        logp, selected_actions, td = self._step(logp, mask, td, **kwargs)

        td.set("action", selected_actions)

        self.actions.append(selected_actions)
        self.logp.append(logp)

        return td


class Greedy(DecodingStrategy):
    name = "greedy"

    def __init__(self, multistart=False, num_starts=None, **kwargs) -> None:
        super().__init__(multistart=multistart, num_starts=num_starts, **kwargs)

    def _step(
        self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Select the action with the highest log probability."""
        # [BS], [BS]
        _, selected = logp.max(1)

        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "infeasible action selected"

        return logp, selected, td


class Sampling(DecodingStrategy):
    name = "sampling"

    def __init__(self, multistart=False, num_starts=None, **kwargs) -> None:
        super().__init__(multistart=multistart, num_starts=num_starts, **kwargs)

    def _step(
        self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Sample an action with a multinomial distribution given by the log probabilities."""
        probs = logp.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)

        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            log.info("Sampled bad values, resampling!")
            selected = probs.multinomial(1).squeeze(1)

        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "infeasible action selected"

        return logp, selected, td


class BeamSearch(DecodingStrategy):
    name = "beam_search"

    def __init__(self, beam_width=None, select_best=True, **kwargs) -> None:
        super().__init__()
        self.beam_width = beam_width
        self.select_best = select_best
        self.parent_beam_logp = None
        self.beam_path = []

    def _step(
        self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        selected, batch_beam_idx = self._make_beam_step(logp)
        # select the correct state representation, logp and mask according to beam parent
        td = td[batch_beam_idx]
        logp = logp[batch_beam_idx]
        mask = mask[batch_beam_idx]

        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "infeasible action selected"

        return logp, selected, td

    def pre_decoder_hook(self, td: TensorDict, env: RL4COEnvBase, **kwargs):
        if self.beam_width is None:
            self.beam_width = get_num_starts(td, env.name)

        # select start nodes. TODO: include first step in beam search as well
        action = self.select_start_nodes_fn(td, env, self.beam_width)

        # Expand td to batch_size * beam_width
        td = batchify(td, self.beam_width)

        td.set("action", action)
        td = env.step(td)["next"]

        log_p = torch.zeros_like(td["action_mask"], device=td.device)
        beam_parent = torch.zeros(log_p.size(0), device=td.device, dtype=torch.int32)

        self.logp.append(log_p)
        self.actions.append(action)
        self.parent_beam_logp = log_p.gather(1, action[..., None])
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
        logp = torch.stack(self.logp, 1)
        assert actions.size(1) == len(
            self.beam_path
        ), "action idx shape and beam path shape dont match"

        # [BS*BW]
        cur_parent = self.beam_path[-1]
        # [BS*BW]
        reversed_aligned_sequences = [actions[:, -1]]
        reversed_aligned_logp = [logp[:, -1]]

        aug_batch_size = actions.size(0)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(actions.device)
        )

        for k in reversed(range(len(self.beam_path) - 1)):
            batch_beam_idx = batch_beam_sequence + cur_parent * batch_size

            reversed_aligned_sequences.append(actions[batch_beam_idx, k])
            reversed_aligned_logp.append(logp[batch_beam_idx, k])
            cur_parent = self.beam_path[k][batch_beam_idx]

        # [BS*BW, seq_len*num_targets]
        actions = torch.stack(list(reversed(reversed_aligned_sequences)), dim=1)
        logp = torch.stack(list(reversed(reversed_aligned_logp)), dim=1)

        return actions, logp

    def _select_best_beam(self, logp, actions, td: TensorDict, env: RL4COEnvBase):
        aug_batch_size = logp.size(0)  # num nodes
        batch_size = aug_batch_size // self.beam_width
        rewards = env.get_reward(td, actions)
        _, idx = torch.cat(rewards.unsqueeze(1).split(batch_size), 1).max(1)
        flat_idx = torch.arange(batch_size, device=rewards.device) + idx * batch_size
        return logp[flat_idx], actions[flat_idx], td[flat_idx], env

    def _make_beam_step(self, logp: torch.Tensor):
        aug_batch_size, num_nodes = logp.shape  # num nodes
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = (
            torch.arange(0, batch_size).repeat(self.beam_width).to(logp.device)
        )

        # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
        log_beam_prob = logp + self.parent_beam_logp  #

        # [BS, num_nodes * BW]
        log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)
        # [BS, BW]
        topk_logp, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1)

        # [BS*BW, 1]
        logp_selected = torch.hstack(torch.unbind(topk_logp, 1)).unsqueeze(1)

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind, 1))

        # since we stack the logprobs from the distinct branches, the indices in
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index

        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        self.parent_beam_logp = logp_selected
        self.beam_path.append(beam_parent)

        return selected, batch_beam_idx
