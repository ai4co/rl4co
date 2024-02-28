import math

from typing import Union

import torch

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.utils import decode_probs
from rl4co.utils.ops import batchify, unbatchify


def forward_logit_attn_eas_lay(
    self, query, key, value, logit_key, mask, softmax_temp=None
):
    """Add layer to the forward pass of logit attention, i.e.
    Single-head attention.
    """
    # Compute inner multi-head attention with no projections.
    heads = self._inner_mha(query, key, value, mask)

    # Add residual for EAS layer if is set
    if getattr(self, "eas_layer", None) is not None:
        heads = heads + self.eas_layer(heads)

    glimpse = self.project_out(heads)

    # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
    # bmm is slightly faster than einsum and matmul
    logits = (
        torch.bmm(glimpse, logit_key.squeeze(1).transpose(-2, -1))
        / math.sqrt(glimpse.size(-1))
    ).squeeze(1)

    # From the logits compute the probabilities by clipping, masking and softmax
    if self.tanh_clipping > 0:
        logits = torch.tanh(logits) * self.tanh_clipping

    if self.mask_logits:
        logits[mask] = float("-inf")

    # Normalize with softmax and apply temperature
    if self.normalize:
        softmax_temp = softmax_temp if softmax_temp is not None else self.softmax_temp
        logits = torch.log_softmax(logits / softmax_temp, dim=-1)

    assert not torch.isnan(logits).any(), "Logits contain NaNs"

    return logits


def forward_eas(
    self,
    td: TensorDict,
    cached_embeds,
    best_solutions,
    iter_count: int = 0,
    env: Union[str, RL4COEnvBase] = None,
    decode_type: str = "multistart_sampling",
    num_starts: int = None,
    softmax_temp: float = None,
    **unused_kwargs,
):
    """Forward pass of the decoder
    Given the environment state and the pre-computed embeddings, compute the logits and sample actions

    Args:
        td: Input TensorDict containing the environment state
        embeddings: Precomputed embeddings for the nodes. Can be already precomputed cached in form of q, k, v and
        env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
            it is more efficient to pass an already instantiated environment each time for fine-grained control
        decode_type: Type of decoding to use. Can be one of:
            - "sampling": sample from the logits
            - "greedy": take the argmax of the logits
            - "multistart_sampling": sample as sampling, but with multi-start decoding
            - "multistart_greedy": sample as greedy, but with multi-start decoding
        num_starts: Number of multi-starts to use. If None, will be calculated from the action mask
        softmax_temp: Temperature for the softmax. If None, default softmax is used from the `LogitAttention` module
        calc_reward: Whether to calculate the reward for the decoded sequence
    """

    # Collect outputs
    outputs = []
    actions = []

    decode_step = 0
    # Multi-start decoding: first action is chosen by ad-hoc node selection
    if num_starts > 1 or "multistart" in decode_type:
        action = (
            self.select_start_nodes_fn(td, env, num_starts=num_starts + 1) % num_starts
        )
        # Append incumbent solutions
        if iter_count > 0:
            action = unbatchify(action, num_starts + 1)
            action[:, -1] = best_solutions[:, decode_step]
            action = action.permute(1, 0).reshape(-1)

        # Expand td to batch_size * (num_starts + 1)
        td = batchify(td, num_starts + 1)

        td.set("action", action)
        td = env.step(td)["next"]
        log_p = torch.zeros_like(
            td["action_mask"], device=td.device
        )  # first log_p is 0, so p = log_p.exp() = 1

        outputs.append(log_p)
        actions.append(action)

    # Main decoding: loop until all sequences are done
    while not td["done"].all():
        decode_step += 1
        log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp, num_starts + 1)

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        action = decode_probs(log_p.exp(), mask, decode_type=decode_type)

        if iter_count > 0:  # append incumbent solutions
            init_shp = action.shape
            action = unbatchify(action, num_starts + 1)
            action[:, -1] = best_solutions[:, decode_step]
            action = action.permute(1, 0).reshape(init_shp)

        td.set("action", action)
        td = env.step(td)["next"]

        # Collect output of step
        outputs.append(log_p)
        actions.append(action)

    outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
    rewards = env.get_reward(td, actions)
    return outputs, actions, td, rewards
