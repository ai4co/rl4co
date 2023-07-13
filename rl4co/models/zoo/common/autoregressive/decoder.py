from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.attention import LogitAttention
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.utils import decode_probs
from rl4co.utils.ops import batchify, select_start_nodes, unbatchify


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class AutoregressiveDecoder(nn.Module):
    """Auto-regressive decoder for constructing solutions for combinatorial optimization problems.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    We additionally include support for multi-starts as it is more efficient to do so in the decoder as we can
    natively perform the attention computation.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward is not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/kaist-silab/rl4co/issues/72.

    Args:
        env_name: environment name to solve
        embedding_dim: Dimension of the embeddings
        num_heads: Number of heads for the attention
        use_graph_context: Whether to use the initial graph context to modify the query
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int,
        num_heads: int,
        use_graph_context: bool = True,
        **logit_attn_kwargs,
    ):
        super().__init__()

        self.env_name = env_name
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.context_embedding = env_context_embedding(
            self.env_name, {"embedding_dim": embedding_dim}
        )
        self.dynamic_embedding = env_dynamic_embedding(
            self.env_name, {"embedding_dim": embedding_dim}
        )
        self.use_graph_context = use_graph_context

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # MHA
        self.logit_attention = LogitAttention(
            embedding_dim, num_heads, **logit_attn_kwargs
        )

    def forward(
        self,
        td: TensorDict,
        embeddings: Tensor,
        env: Union[str, RL4COEnvBase] = None,
        decode_type: str = "sampling",
        num_starts: int = None,
        softmax_temp: float = None,
        calc_reward: bool = True,
    ) -> Tuple[Tensor, Tensor, TensorDict]:
        """Forward pass of the decoder
        Given the environment state and the pre-computed embeddings, compute the logits and sample actions

        Args:
            td: Input TensorDict containing the environment state
            embeddings: Precomputed embeddings for the nodes
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            decode_type: Type of decoding to use. Can be one of:
                - "sampling": sample from the logits
                - "greedy": take the argmax of the logits
                - "multistart_sampling": sample as sampling, but with multi-start decoding
                - "multistart_greedy": sample as greedy, but with multi-start decoding
            num_starts: Number of multi-starts to use. If None, no multi-start decoding is used
            softmax_temp: Temperature for the softmax. If None, default softmax is used from the `LogitAttention` module
            calc_reward: Whether to calculate the reward for the decoded sequence

        Returns:
            outputs: Tensor of shape (batch_size, seq_len, num_nodes) containing the logits
            actions: Tensor of shape (batch_size, seq_len) containing the sampled actions
            td: TensorDict containing the environment state after decoding
        """

        # Greedy multi-start decoding if num_starts > 1
        num_starts = 0 if num_starts is None else num_starts
        assert not (
            "multistart" in decode_type and num_starts <= 1
        ), "Multi-start decoding requires `num_starts` > 1"

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute_cache(embeddings, num_starts=num_starts)

        # Collect outputs
        outputs = []
        actions = []

        # Instantiate environment if needed
        if isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if num_starts > 1 or "multistart" in decode_type:
            action = select_start_nodes(td, num_starts, env)

            # Expand td to batch_size * num_starts
            td = batchify(td, num_starts)

            td.set("action", action)
            td = env.step(td)["next"]
            log_p = torch.zeros_like(
                td["action_mask"], device=td.device
            )  # first log_p is 0, so p = log_p.exp() = 1

            outputs.append(log_p)
            actions.append(action)

        # Main decoding
        while not td["done"].all():
            log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp, num_starts)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            action = decode_probs(log_p.exp(), mask, decode_type=decode_type)

            td.set("action", action)
            td = env.step(td)["next"]

            # Collect output of step
            outputs.append(log_p)
            actions.append(action)

        outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        return outputs, actions, td

    def _precompute_cache(self, embeddings: Tensor, num_starts: int = 0):
        """Compute the cached embeddings for the attention

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of multi-starts to use. If 0, no multi-start decoding is used
        """

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = unbatchify(
                batchify(self.project_fixed_context(embeddings.mean(1)), num_starts),
                num_starts,
            )
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

        return cached_embeds

    def _get_log_p(
        self,
        cached: PrecomputedCache,
        td: TensorDict,
        softmax_temp: float = None,
        num_starts: int = 0,
    ):
        """Compute the log probabilities of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            softmax_temp: Temperature for the softmax
            num_starts: Number of starts for the multi-start decoding
        """

        # Unbatchify to [batch_size, num_starts, ...]. Has no effect if num_starts = 0
        td_unbatch = unbatchify(td, num_starts)
        step_context = self.context_embedding(cached.node_embeddings, td_unbatch)
        glimpse_q = step_context + cached.graph_context
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q

        # Compute keys and values for the nodes
        (
            glimpse_key_dynamic,
            glimpse_val_dynamic,
            logit_key_dynamic,
        ) = self.dynamic_embedding(td_unbatch)
        glimpse_k = cached.glimpse_key + glimpse_key_dynamic
        glimpse_v = cached.glimpse_val + glimpse_val_dynamic
        logit_k = cached.logit_key + logit_key_dynamic

        # Get the mask
        mask = ~td_unbatch["action_mask"]

        # Compute logits
        log_p = self.logit_attention(
            glimpse_q, glimpse_k, glimpse_v, logit_k, mask, softmax_temp
        )

        # Now we need to reshape the logits and log_p to [batch_size*num_starts, num_nodes]
        # Note that rearranging order is important here
        log_p = rearrange(log_p, "b s l -> (s b) l") if num_starts > 1 else log_p
        mask = rearrange(mask, "b s l -> (s b) l") if num_starts > 1 else mask
        return log_p, mask
