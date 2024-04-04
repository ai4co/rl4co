from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import Normalization


class MixedScoresSDPA(nn.Module):
    def __init__(
        self,
        num_heads: int,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        super().__init__()
        mix_W1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, 2, mixer_hidden_dim)
        )
        mix_b1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, mixer_hidden_dim)
        )
        self.mix_W1 = nn.Parameter(mix_W1)
        self.mix_b1 = nn.Parameter(mix_b1)

        mix_W2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, mixer_hidden_dim, 1)
        )
        mix_b2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, 1)
        )
        self.mix_W2 = nn.Parameter(mix_W2)
        self.mix_b2 = nn.Parameter(mix_b2)

    def forward(self, q, k, v, dmat, attn_mask=None, dropout_p=0.0, is_causal=False):
        """Scaled Dot-Product Attention with MatNet Scores Mixer"""
        b, m, n = dmat.shape
        # Check for causal and attn_mask conflict
        if is_causal and attn_mask is not None:
            raise ValueError("Cannot set both is_causal and attn_mask")

        # Calculate scaled dot product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        mix_attn_scores = torch.stack(
            [attn_scores, dmat[:, None, :, :].expand(b, self.num_heads, m, n)], dim=-1
        )  # [b, h, m, n, 2]

        attn_scores = (
            (
                torch.matmul(
                    F.relu(
                        torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
                        + self.mix_b1[None, None, :, None, :]
                    ),
                    self.mix_W2,
                )
                + self.mix_b2[None, None, :, None, :]
            )
            .transpose(1, 2)
            .squeeze(-1)
        )  # [b, h, m, n]

        # Apply the provided attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_scores += attn_mask

        # Apply causal mask
        if is_causal:
            s, l_ = attn_scores.size(-2), attn_scores.size(-1)
            mask = torch.triu(torch.ones((s, l_), device=attn_scores.device), diagonal=1)
            attn_scores.masked_fill_(mask.bool(), float("-inf"))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # Compute the weighted sum of values
        return torch.matmul(attn_weights, v)


class MatNetCrossMHA(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        bias: bool = False,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert (
            self.embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        self.head_dim = self.embedding_dim // num_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.Wkv = nn.Linear(embedding_dim, 2 * embedding_dim, bias=bias)

        self.sdpa = MixedScoresSDPA(
            num_heads=num_heads,
            mixer_hidden_dim=mixer_hidden_dim,
            mix1_init=mix1_init,
            mix2_init=mix2_init,
        )

        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def forward(self, q_input, kv_input, dmat):
        """

        Args:
            q_input (Tensor): [b, m, d]
            kv_input (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Tensor: [b, m, d]
        """

        q = rearrange(
            self.Wq(q_input), "b m (h d) -> b h m d", h=self.num_heads
        )  # [b, h, m, d]
        k, v = rearrange(
            self.Wkv(kv_input), "b n (two h d) -> two b h n d", two=2, h=self.num_heads
        ).unbind(
            dim=0
        )  # [b, h, n, d]

        out = self.sdpa(q, k, v, dmat)
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class MatNetMHA(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.row_encoding_block = MatNetCrossMHA(embedding_dim, num_heads, bias)
        self.col_encoding_block = MatNetCrossMHA(embedding_dim, num_heads, bias)

    def forward(self, row_emb, col_emb, dmat):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """

        updated_row_emb = self.row_encoding_block(row_emb, col_emb, dmat)
        updated_col_emb = self.col_encoding_block(
            col_emb, row_emb, dmat.transpose(-2, -1)
        )
        return updated_row_emb, updated_col_emb


class MatNetMHALayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        bias: bool = False,
        feed_forward_hidden: int = 512,
        normalization: Optional[str] = "instance",
    ):
        super().__init__()
        self.MHA = MatNetMHA(embedding_dim, num_heads, bias)

        self.F_a = nn.ModuleDict(
            {
                "norm1": Normalization(embedding_dim, normalization),
                "ffn": nn.Sequential(
                    nn.Linear(embedding_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embedding_dim),
                ),
                "norm2": Normalization(embedding_dim, normalization),
            }
        )

        self.F_b = nn.ModuleDict(
            {
                "norm1": Normalization(embedding_dim, normalization),
                "ffn": nn.Sequential(
                    nn.Linear(embedding_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embedding_dim),
                ),
                "norm2": Normalization(embedding_dim, normalization),
            }
        )

    def forward(self, row_emb, col_emb, dmat):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """

        row_emb_out, col_emb_out = self.MHA(row_emb, col_emb, dmat)

        row_emb_out = self.F_a["norm1"](row_emb + row_emb_out)
        row_emb_out = self.F_a["norm2"](row_emb_out + self.F_a["ffn"](row_emb_out))

        col_emb_out = self.F_b["norm1"](col_emb + col_emb_out)
        col_emb_out = self.F_b["norm2"](col_emb_out + self.F_b["ffn"](col_emb_out))
        return row_emb_out, col_emb_out


class MatNetMHANetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        bias: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MatNetMHALayer(
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    feed_forward_hidden=feed_forward_hidden,
                    normalization=normalization,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, row_emb, col_emb, dmat):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, dmat)
        return row_emb, col_emb


class MatNetEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 5,
        normalization: str = "instance",
        feed_forward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = None,
        bias: bool = False,
    ):
        super().__init__()

        if init_embedding is None:
            init_embedding = env_init_embedding(
                "matnet", {"embedding_dim": embedding_dim, **init_embedding_kwargs}
            )

        self.init_embedding = init_embedding
        self.net = MatNetMHANetwork(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization=normalization,
            feed_forward_hidden=feed_forward_hidden,
            bias=bias,
        )

    def forward(self, td):
        row_emb, col_emb, dmat = self.init_embedding(td)
        row_emb, col_emb = self.net(row_emb, col_emb, dmat)

        embedding = (row_emb, col_emb)
        init_embedding = None
        return embedding, init_embedding  # match output signature for the AR policy class
