from dataclasses import dataclass
from einops import rearrange
import math

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

from ncobench.models.nn.attention import flash_attn_wrapper
from ncobench.models.components.am.context import env_context
from ncobench.models.components.am.embeddings import env_dynamic_embedding
from ncobench.models.components.am.utils import decode_probs
from ncobench.utils import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor
    

class LogitAttention(nn.Module):
    """Calculate logits given query, key and value and logit key
    If we use Flash Attention, then we automatically move to fp16 for inner computations
    Note: with Flash Attention, masking is not supported

    Perform the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key
        4. Normalize and mask
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalize=True,
        force_flash_attn=False,
    ):
        super(LogitAttention, self).__init__()
        self.n_heads = n_heads
        self.mask_logits = mask_logits
        self.mask_inner = mask_inner
        self.tanh_clipping = tanh_clipping
        self.normalize = normalize
        self.force_flash_attn = force_flash_attn

        if force_flash_attn and mask_inner:
            log.warn(
                "Flash Attention does not support masking, force_flash_attn will only be used for fp16"
            )

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, logit_key, mask):
        # Compute inner multi-head attention with no projections
        heads = self._inner_mha(query, key, value, mask)
        glimpse = self.project_out(heads)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = torch.bmm(
            glimpse.squeeze(1), logit_key.squeeze(1).transpose(-2, -1)
        ) / math.sqrt(glimpse.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        if self.mask_logits:
            logits[mask] = float("-inf")

        if self.normalize:
            logits = torch.log_softmax(logits, dim=-1)

        assert not torch.isnan(logits).any()

        return logits

    def _inner_mha(self, query, key, value, mask):
        query = rearrange(query, "b 1 (h s) -> b h 1 s", h=self.n_heads)
        mask = ~mask.unsqueeze(1) if self.mask_inner else None
        heads = self.flash_attn_wrapper(
            scaled_dot_product_attention, query, key, value, attn_mask=mask
        )
        heads = rearrange(heads, "b h 1 g -> b 1 1 (h g)", h=self.n_heads)
        return heads

    def _make_heads(self, v):
        return rearrange(v, "b 1 g (h s) -> b h g s", h=self.n_heads)

    flash_attn_wrapper = flash_attn_wrapper


class Decoder(nn.Module):
    def __init__(self, env, embedding_dim, n_heads, **logit_attn_kwargs):
        super(Decoder, self).__init__()

        self.env = env
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        assert embedding_dim % n_heads == 0

        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        self.context = env_context(self.env.name, {"context_dim": step_context_dim})
        self.dynamic_embedding = env_dynamic_embedding(
            self.env.name, {"embedding_dim": embedding_dim}
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(
            step_context_dim, embedding_dim, bias=False
        )

        # MHA
        self.logit_attention = LogitAttention(
            embedding_dim, n_heads, **logit_attn_kwargs
        )

    def forward(self, td, embeddings, decode_type="sampling"):
        outputs = []
        actions = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute(embeddings)

        while not td[
            "done"
        ].any():  # NOTE: here we suppose all the batch is done at the same time
            log_p, mask = self._get_log_p(cached_embeds, td)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            action = decode_probs(
                log_p.exp().squeeze(1), mask.squeeze(1), decode_type=decode_type
            )

            td.set("action", action[:, None])
            td = self.env.step(td)["next"]

            # Collect output of step
            outputs.append(log_p.squeeze(1))
            actions.append(action)

        outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        td.set("reward", self.env.get_reward(td["observation"], actions))
        return outputs, actions, td

    def _precompute(self, embeddings):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # Organize in a TensorDict for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=self.project_fixed_context(graph_embed)[:, None, :],
            glimpse_key=self.logit_attention._make_heads(glimpse_key_fixed),
            glimpse_val=self.logit_attention._make_heads(glimpse_val_fixed),
            logit_key=logit_key_fixed,
        )

        return cached_embeds

    def _get_log_p(self, cached, td):
        context = self.context(cached.node_embeddings, td)
        step_context = self.project_step_context(context)  # [batch, 1, embed_dim]

        query = cached.graph_context + step_context  # [batch, 1, embed_dim]

        # Compute keys and values for the nodes
        # glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(cached, td['observation'])
        (
            glimpse_key_dynamic,
            glimpse_val_dynamic,
            logit_key_dynamic,
        ) = self.dynamic_embedding(td["observation"])
        glimpse_key = cached.glimpse_key + glimpse_key_dynamic
        glimpse_key = cached.glimpse_val + glimpse_val_dynamic
        logit_key = cached.logit_key + logit_key_dynamic

        # Get the mask
        mask = ~td["action_mask"]

        # Compute logits
        log_p = self.logit_attention(query, glimpse_key, glimpse_key, logit_key, mask)

        return log_p, mask
