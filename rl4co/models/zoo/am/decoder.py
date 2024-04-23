from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class AttentionModelDecoder(AutoregressiveDecoder):
    """

    # TODO
    Auto-regressive decoder for constructing solutions for combinatorial optimization problems.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    We additionally include support for multi-starts as it is more efficient to do so in the decoder as we can
    natively perform the attention computation.



    Args:
        # TODO
        env_name: environment name to solve
        embed_dim: Dimension of the embeddings
        num_heads: Number of heads for the attention
        use_graph_context: Whether to use the initial graph context to modify the query
        context_embedding: Module to compute the context embedding. If None, the default is used
        dynamic_embedding: Module to compute the dynamic embedding. If None, the default is used
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
        self.pointer = PointerAttention(
            embed_dim,
            num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        self.project_node_embeddings = nn.Linear(
            embed_dim, 3 * embed_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        # Get precomputed (cached) embeddings
        node_embeds_cache, graph_context_cache = (
            cached.node_embeddings,
            cached.graph_context,
        )
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )  # [B, N, H]
        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            glimpse_k_stat = batchify(glimpse_k_stat, num_starts)
            glimpse_v_stat = batchify(glimpse_v_stat, num_starts)
            logit_k_stat = batchify(logit_k_stat, num_starts)
            node_embeds_cache = batchify(node_embeds_cache, num_starts)
            graph_context_cache = (
                batchify(graph_context_cache, num_starts)
                if isinstance(graph_context_cache, Tensor)
                else graph_context_cache
            )
        elif num_starts > 1:
            td = unbatchify(td, num_starts)
            if isinstance(graph_context_cache, Tensor):
                # add a dimension for num_starts (will automatically be broadcasted during addition)
                graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        glimpse_q = (
            glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        )  # add seq_len dim if not present

        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        # Compute logits
        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        return td, env, self._precompute_cache(embeddings, num_starts)

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )
