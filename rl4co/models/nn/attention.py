import math

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.utils import get_pylogger

log = get_pylogger(__name__)


def scaled_dot_product_attention_simple(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
        "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
    )
    scaled_dot_product_attention = scaled_dot_product_attention_simple


class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
        sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout

        # Default to `scaled_dot_product_attention` if `sdpa_fn` is not provided
        if sdpa_fn is None:
            sdpa_fn = scaled_dot_product_attention
        self.sdpa_fn = sdpa_fn

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class LogitAttention(nn.Module):
    """Calculate logits given query, key and value and logit key.

    Note:
        With Flash Attention, masking is not supported

    Perform the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key
        4. Normalize and mask

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        tanh_clipping: tanh clipping value
        mask_inner: whether to mask inner attention
        mask_logits: whether to mask logits
        normalize: whether to normalize logits
        softmax_temp: softmax temperature
        linear_bias: whether to use bias in linear projection
        sdp_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        tanh_clipping: float = 10.0,
        mask_inner: bool = True,
        mask_logits: bool = True,
        normalize: bool = True,
        softmax_temp: float = 1.0,
        linear_bias: bool = False,
        sdp_fn=scaled_dot_product_attention,
    ):
        super(LogitAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_logits = mask_logits
        self.mask_inner = mask_inner
        self.tanh_clipping = tanh_clipping
        self.normalize = normalize
        self.softmax_temp = softmax_temp

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.sdp_fn = sdp_fn

    def forward(self, query, key, value, logit_key, mask, softmax_temp=None):
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, mask)
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

    def _inner_mha(self, query, key, value, mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)

        if self.mask_inner:
            # need to invert mask: (N L S) -> (N 1 L S)
            attn_mask = (
                ~mask.unsqueeze(1) if mask.ndim == 3 else ~mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None

        heads = self.sdp_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)
