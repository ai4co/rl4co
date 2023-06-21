import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.utils import get_pylogger

log = get_pylogger(__name__)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention"
    )

    def scaled_dot_product_attention(
        Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
        if scale is None:
            scale = Q.size(-1) ** -0.5  # scale factor
        # compute the attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # apply causal masking if required
        if is_causal:
            mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        # apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        # compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        # apply dropout
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        # compute the weighted sum of values
        return torch.matmul(attn_probs, V)


def flash_attn_wrapper(self, func, *args, **kwargs):
    """Wrapper for flash attention to automatically cast to fp16 if needed"""
    if self.force_flash_attn and args[0].is_cuda:
        original_dtype = args[0].dtype
        args = [arg.half() for arg in args if isinstance(arg, torch.Tensor)]
        out = func(*args, **kwargs)
        return out.to(original_dtype)
    else:
        return func(*args, **kwargs)


class NativeFlashMHA(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        force_flash_attn=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.force_flash_attn = force_flash_attn
        self.attention_dropout = attention_dropout

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
        out = self.flash_attn_wrapper(
            scaled_dot_product_attention,
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))

    flash_attn_wrapper = flash_attn_wrapper


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module following Kool et al. (2019)"""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hdim = embed_dim // num_heads

        self.norm_factor = 1 / math.sqrt(self.hdim)  # See Attention is all you need

        self.Wq = nn.Parameter(torch.Tensor(num_heads, embed_dim, self.hdim))
        self.Wk = nn.Parameter(torch.Tensor(num_heads, embed_dim, self.hdim))
        self.Wv = nn.Parameter(torch.Tensor(num_heads, embed_dim, self.hdim))

        self.Wout = nn.Parameter(torch.Tensor(num_heads, self.hdim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """q: queries (batch_size, n_query, input_dim)
        h: data (batch_size, graph_size, input_dim)
        mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        """

        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # Last dimension can be different for keys and values
        shp = (self.num_heads, batch_size, graph_size, -1)
        shp_q = (self.num_heads, batch_size, n_query, -1)

        # Calculate queries, (num_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.Wq).view(shp_q)
        # Calculate keys and values (num_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.Wk).view(shp)
        V = torch.matmul(hflat, self.Wv).view(shp)

        # Calculate compatibility (num_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = float("-inf")  # -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.num_heads * self.hdim),
            self.Wout.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)

        return out


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
        num_heads,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalize=True,
        softmax_temp=1.0,
        force_flash_attn=False,
    ):
        super(LogitAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_logits = mask_logits
        self.mask_inner = mask_inner
        self.tanh_clipping = tanh_clipping
        self.normalize = normalize
        self.softmax_temp = softmax_temp
        self.force_flash_attn = force_flash_attn

        if force_flash_attn and mask_inner:
            log.warn(
                "Flash Attention does not support masking, force_flash_attn will only be used for fp16"
            )

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

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

        heads = self.flash_attn_wrapper(
            scaled_dot_product_attention, q, k, v, attn_mask=attn_mask
        )
        return rearrange(heads, "b h n g -> b n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "b g (h s) -> b h g s", h=self.num_heads)

    flash_attn_wrapper = flash_attn_wrapper
