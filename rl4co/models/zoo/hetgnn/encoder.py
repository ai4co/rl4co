import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum
from torch import Tensor

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import Normalization


class HetGNNLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.self_attn = nn.Parameter(torch.rand(size=(embed_dim, 1), dtype=torch.float))
        self.cross_attn = nn.Parameter(torch.rand(size=(embed_dim, 1), dtype=torch.float))
        self.edge_attn = nn.Parameter(torch.rand(size=(embed_dim, 1), dtype=torch.float))
        self.activation = nn.ReLU()
        self.scale = 1 / math.sqrt(embed_dim)

    def forward(
        self, self_emb: Tensor, other_emb: Tensor, edge_emb: Tensor, edges: Tensor
    ):
        bs, n_rows, _ = self_emb.shape

        # concat operation embeddings and o-m edge features (proc times)
        # Calculate attention coefficients
        er = einsum(self_emb, self.self_attn, "b m e, e one -> b m") * self.scale
        ec = einsum(other_emb, self.cross_attn, "b o e, e one -> b o") * self.scale
        ee = einsum(edge_emb, self.edge_attn, "b m o e, e one -> b m o") * self.scale

        # element wise multiplication similar to broadcast column logits over rows with masking
        ec_expanded = einsum(edges, ec, "b m o, b o -> b m o")
        # element wise multiplication similar to broadcast row logits over cols with masking
        er_expanded = einsum(edges, er, "b m o, b m -> b m o")

        # adding the projections of different node types and edges together (equivalent to first concat and then project)
        # (bs, n_rows, n_cols)
        cross_logits = self.activation(ec_expanded + ee + er_expanded)

        # (bs, n_rows, 1)
        self_logits = self.activation(er + er).unsqueeze(-1)

        # (bs, n_ma, n_ops + 1)
        mask = torch.cat(
            (
                edges == 1,
                torch.full(
                    size=(bs, n_rows, 1),
                    dtype=torch.bool,
                    fill_value=True,
                    device=edges.device,
                ),
            ),
            dim=-1,
        )

        # (bs, n_ma, n_ops + 1)
        all_logits = torch.cat((cross_logits, self_logits), dim=-1)
        all_logits[~mask] = -torch.inf
        attn_scores = F.softmax(all_logits, dim=-1)
        # (bs, n_ma, n_ops)
        cross_attn_scores = attn_scores[..., :-1]
        # (bs, n_ma, 1)
        self_attn_scores = attn_scores[..., -1].unsqueeze(-1)

        # augment column embeddings with edge features, (bs, r, c, e)
        other_emb_aug = edge_emb + other_emb.unsqueeze(-3)
        cross_emb = einsum(cross_attn_scores, other_emb_aug, "b m o, b m o e -> b m e")
        self_emb = self_emb * self_attn_scores
        # (bs, n_ma, emb_dim)
        hidden = torch.sigmoid(cross_emb + self_emb)
        return hidden


class HetGNNBlock(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.norm1 = Normalization(embed_dim, normalization="batch")
        self.norm2 = Normalization(embed_dim, normalization="batch")
        self.hgnn1 = HetGNNLayer(embed_dim)
        self.hgnn2 = HetGNNLayer(embed_dim)

    def forward(self, x1, x2, edge_emb, edges):
        h1 = self.hgnn1(x1, x2, edge_emb, edges)
        h1 = self.norm1(h1 + x1)

        h2 = self.hgnn2(x2, x1, edge_emb.transpose(1, 2), edges.transpose(1, 2))
        h2 = self.norm2(h2 + x2)

        return h1, h2


class HetGNNEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        init_embedding=None,
        edge_key: str = "ops_ma_adj",
        edge_weights_key: str = "proc_times",
        linear_bias: bool = False,
    ) -> None:
        super().__init__()

        if init_embedding is None:
            init_embedding = env_init_embedding("fjsp", {"embed_dim": embed_dim})
        self.init_embedding = init_embedding

        self.edge_key = edge_key
        self.edge_weights_key = edge_weights_key

        self.num_layers = num_layers
        self.layers = nn.ModuleList([HetGNNBlock(embed_dim) for _ in range(num_layers)])

    def forward(self, td):
        edges = td[self.edge_key]
        bs, n_rows, n_cols = edges.shape
        row_emb, col_emb, edge_emb = self.init_embedding(td)
        assert row_emb.size(1) == n_rows, "incorrect number of row embeddings"
        assert col_emb.size(1) == n_cols, "incorrect number of column embeddings"

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, edge_emb, edges)

        return (row_emb, col_emb), None
