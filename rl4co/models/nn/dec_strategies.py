import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
import warnings
from typing import Tuple
from collections import defaultdict
from dataclasses import dataclass, fields, replace


from rl4co.envs import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import batchify, get_num_starts, select_start_nodes, unbatchify


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
            f"Unknown environment name '{decoding_strategy}'. Available dynamic embeddings: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    if "multistart" in decoding_strategy:
        config["multistart"] = True

    return strategy_registry.get(decoding_strategy, Sampling)(**config)


class DecodingStrategy(nn.Module):

    name = ...

    def __init__(self, *args, multistart=False, **kwargs) -> None:
        super().__init__()
        self.actions = []
        self.logp = []
        self.multistart = multistart


    def _step(self, logp: torch.Tensor, td: TensorDict, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must be implemented by subclass")
    

    def _pre_step_hook(self, logp: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs):
        # probs = logp.exp()
        # assert (probs == probs).all(), "Probs should not contain any nans"
        assert not logp.isinf().all(1).any()  # This should do the same but without doing the .exp transform
        # assert ~logp.isinf().all(1).all() 

        return logp, td


    def _post_step_hook(self, selected: torch.Tensor, mask: torch.Tensor, td: TensorDict, **kwargs):

        assert not mask.gather(
            1, selected.unsqueeze(-1)
        ).data.any(), "infeasible action selected"

        return selected, td
    

    def pre_hook(self, td: TensorDict, env: RL4COEnvBase, num_starts: int = None):

        # Multi-start decoding. If num_starts is None, we use the number of actions in the action mask
        if self.multistart:
            if num_starts is None:
                self.num_starts = get_num_starts(td, env.name)
        else:
            if num_starts is not None:
                if num_starts > 1:
                    log.warn(
                        f"num_starts={num_starts} is ignored for decode_type={self.name}"
                    )

            self.num_starts = 0

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.num_starts > 1:
            action = select_start_nodes(td, env, self.num_starts)

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

    def post_hook(self, td, env):

        assert (
            len(self.logp) > 0
        ), "No outputs were collected because all environments were done. Check your initial state"

        return torch.stack(self.logp, 1), torch.stack(self.actions, 1), td, env
    
    
    def step(self, 
             logp: torch.Tensor, 
             mask: torch.Tensor,
             td: TensorDict, 
             **kwargs) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        
        logp, td = self._pre_step_hook(logp, mask, td, **kwargs)        
        selected_actions, td = self._step(logp, mask, td, **kwargs)
        selected_actions, td = self._post_step_hook(selected_actions, mask, td, **kwargs) 

        td.set("action", selected_actions)

        self.actions.append(selected_actions)
        self.logp.append(logp)

        return td 


class Greedy(DecodingStrategy):

    name = "greedy"

    def __init__(self, *args, multistart=False, **kwargs) -> None:
        super().__init__()
        self.multistart = multistart


    def _step(self, 
              logp: torch.Tensor, 
              mask: torch.Tensor, 
              td: TensorDict, 
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        
        # [BS], [BS]
        _, selected = logp.max(1)

        return selected, td


class Sampling(DecodingStrategy):

    name = "sampling"

    def __init__(self, *args, multistart=False, **kwargs) -> None:
        super().__init__()
        self.multistart = multistart

    def _step(self, 
              logp: torch.Tensor, 
              mask: torch.Tensor,
              td: TensorDict, 
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        
        probs = logp.exp()
        selected = torch.multinomial(probs, 1).squeeze(1)

        while mask.gather(1, selected.unsqueeze(-1)).data.any():
            log.info("Sampled bad values, resampling!")
            selected = probs.multinomial(1).squeeze(1)

        return selected, td


class BeamSearch(DecodingStrategy):

    name = "beam_search"

    def __init__(self, beam_width, select_best=True, *args, **kwargs) -> None:
        super().__init__()
        self.beam_width = beam_width
        self.select_best = select_best
        self.step_num = 0
        self.log_beam_probs = []
        self.beam_path = []
        if beam_width <= 1:
            warnings.warn("Beam width is <= 1 in Beam search. This might not be what you want")

    # def setup_state(self, state: StateMSVRP):
    #     return state.repeat(self.beam_width)

    def _step(self, 
              probs: torch.Tensor, 
              mask: torch.Tensor, 
              td: TensorDict, 
              **kwargs) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:

        selected, batch_beam_idx = self._make_beam_step(probs)
        # first select the correct state representation according to beam parent
        td = td[batch_beam_idx] 

        self.step_num += 1

        return selected, td

    def _backtrack(self):

        # [BS*BW, seq_len*num_targets]
        actions = torch.stack(self.actions, 1)
        # [BS*BW, seq_len*num_targets]
        logp = torch.stack(self.logp, 1)
        assert actions.size(1) == len(self.beam_path), "action idx shape and beam path shape dont match"

        # [BS*BW]
        cur_parent = self.beam_path[-1]
        # [BS*BW]
        reversed_aligned_sequences = [actions[:, -1]]
        reversed_aligned_logp = [logp[:, -1]]

        aug_batch_size = actions.size(0)
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(actions.device)

        for k in reversed(range(len(self.beam_path)-1)):

            batch_beam_idx = batch_beam_sequence + cur_parent * batch_size 

            reversed_aligned_sequences.append(actions[batch_beam_idx, k])
            reversed_aligned_logp.append(logp[batch_beam_idx, k])
            cur_parent = self.beam_path[k][batch_beam_idx]

        # [BS*BW, seq_len*num_targets]
        actions = torch.stack(list(reversed(reversed_aligned_sequences)), dim=1)
        logp = torch.stack(list(reversed(reversed_aligned_logp)), dim=1)


        return actions, logp
    

    def _select_best_beam(self, probs, actions, td: TensorDict, env: RL4COEnvBase):

        aug_batch_size = probs.size(0)  # num nodes (with depot)
        batch_size = aug_batch_size // self.beam_width

        costs = env.get_reward(td, actions)
        _, idx = torch.cat(costs.unsqueeze(1).split(batch_size), 1).min(1)
        flat_idx = torch.arange(batch_size, device=costs.device) + idx * batch_size
        return probs[flat_idx], actions[flat_idx], td[flat_idx]
    

    def post_hook(self, td, env):
        # [BS*BW, seq_len]
        aligned_sequences, aligned_probs = self._backtrack()

        if self.select_best and not self.training:
            return self._select_best_beam(aligned_probs, aligned_sequences, td, env)
        else:
            return aligned_probs, aligned_sequences, td, env
        

    def _fill_up_beams(self, topk_ind, topk_logp, log_beam_prob):
        """There may be cases where there are less valid options than the specified beam width. This might not be a problem at 
        the start of the beam search, since a few valid options can quickly grow to a lot more options  (if each valid option
        splits up in two more options we have 2^x growth). However, there may also be cases in small instances where simply
        too few options exist. We define these cases when every beam parent has only one valid child and the sum of valid child
        nodes is less than the beam width. In these cases we fill the missing child nodes by duplicating the valid ones.
        
        Moreover, in early phases of the algorithm we may choose invalid nodes to fill the beam. We hardcode these options to
        remain in the depot. These options get filtered out in later phases of the beam search since they have a logprob of -inf

        params:
        - topk_ind
        - topk_logp
        -log_beam_prob_hat [BS, num_nodes * beam_width]
        """
        if self.step_num > 0:

            bs = topk_ind.size(0)
            # [BS, num_nodes, beam_width]
            avail_opt_per_beam = torch.stack(log_beam_prob.split(bs), -1).gt(-torch.inf).sum(1)

            invalid = torch.logical_and(avail_opt_per_beam.le(1).all(1), avail_opt_per_beam.sum(1) < self.beam_width)
            if invalid.any():
                mask = topk_logp[invalid].isinf()
                new_prob, new_ind = topk_logp[invalid].max(1)
                new_prob_exp = new_prob[:,None].expand(-1, self.beam_width)
                new_ind_exp = topk_ind[invalid, new_ind][:,None].expand(-1, self.beam_width)
                topk_logp[invalid] = torch.where(mask, new_prob_exp, topk_logp[invalid])
                topk_ind[invalid] = torch.where(mask, new_ind_exp, topk_ind[invalid])

        # infeasible beam may remain in depot. Beam will be discarded anyway in next round
        topk_ind[topk_logp.eq(-torch.inf)] = 0

        return topk_ind, topk_logp


    def _make_beam_step(self, probs: torch.Tensor):

        aug_batch_size, num_nodes = probs.shape  # num nodes
        batch_size = aug_batch_size // self.beam_width
        batch_beam_sequence = torch.arange(0, batch_size).repeat(self.beam_width).to(probs.device)

        # do log transform in order to avoid that impossible actions are chosen in the beam
        # [BS*BW, num_nodes]
        logp = probs.clone().log()

        if self.step_num == 0:
            # [BS, num_nodes]
            log_beam_prob_hat = logp
            log_beam_prob_hstacked = log_beam_prob_hat[:batch_size]

            if num_nodes < self.beam_width:
                # pack some artificial nodes onto logp
                dummy = torch.full((batch_size, (self.beam_width-num_nodes)), -torch.inf, device=probs.device)
                log_beam_prob_hstacked = torch.hstack((log_beam_prob_hstacked, dummy))

            # [BS, BW]
            topk_logp, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1, sorted=True)

        else:
            # determine the rank of every action per beam (descending order)
            ranks = torch.argsort(torch.argsort(logp, dim=1, descending=True), dim=1)
            # use the rank as penalty so as to promote the best option per beam
            # [BS*BW, num_nodes] + [BS*BW, 1] -> [BS*BW, num_nodes]
            log_beam_prob = logp + self.log_beam_probs[-1].unsqueeze(1) 

            # [BS, num_nodes * BW]
            log_beam_prob_hstacked = torch.cat(log_beam_prob.split(batch_size), dim=1)
            # [BS, BW]
            # _, topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1)
            # NOTE: for testing purposes
            topk_ind = torch.topk(log_beam_prob_hstacked, self.beam_width, dim=1, sorted=True)[1].sort(1)[0]
            # we do not want to keep track of the penalty value, therefore discard it here
            topk_logp = torch.cat(log_beam_prob.split(batch_size), dim=1).gather(1, topk_ind)

        topk_ind, topk_logp = self._fill_up_beams(topk_ind, topk_logp, log_beam_prob)

        # [BS*BW, 1]
        logp_selected = torch.hstack(torch.unbind(topk_logp,1))

        # [BS*BW, 1]
        topk_ind = torch.hstack(torch.unbind(topk_ind,1)) 

        # since we stack the logprobs from the distinct branches, the indices in 
        # topk dont correspond to node indices directly and need to be translated
        selected = topk_ind % num_nodes  # determine node index

        # calc parent this branch comes from
        beam_parent = (topk_ind // num_nodes).int()

        # extract the correct representations from "batch". Consider instances with 10 nodes and 
        # a beam width of 3. If for the first problem instance the corresponding pointers are 1, 
        # 5 and 15, we know that the first two branches come from the first root hypothesis, while 
        # the latter comes from the second hypothesis. Consequently, the first two branches use the
        # first representation of that instance while the latter uses its second representation
        batch_beam_idx = batch_beam_sequence + beam_parent * batch_size

        self.beam_path.append(beam_parent)

        return selected, batch_beam_idx