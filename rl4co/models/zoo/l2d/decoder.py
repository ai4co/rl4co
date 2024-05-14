import torch
import torch.nn as nn

from einops import einsum

from rl4co.models.common.constructive.autoregressive import AutoregressiveDecoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.models.nn.mlp import MLP
from rl4co.utils.ops import batchify, gather_by_index


class JSSPActor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, hidden_layers: int = 2) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=2 * embed_dim,
            output_dim=1,
            num_neurons=[hidden_dim] * hidden_layers,
            hidden_act="Tanh",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )
        self.dummy = nn.Parameter(torch.rand(2 * embed_dim))

    def forward(self, td, ma_emb, op_emb):
        bs = td.size(0)
        # (bs, n_j)
        next_op = td["next_op"]
        job_emb = gather_by_index(op_emb, next_op, dim=1)
        ma_emb_per_op = einsum(td["ops_ma_adj"], ma_emb, "b m o, b m e -> b o e")
        ma_emb_per_job = gather_by_index(ma_emb_per_op, next_op, dim=1)
        job_emb = torch.cat((job_emb, ma_emb_per_job), dim=2)

        no_ops = self.dummy[None, None].expand(bs, 1, -1)  # [bs, 1, 2*emb_dim]
        # [bs, 1 + num_j, 2*emb_dim]
        all_actions = torch.cat((no_ops, job_emb), 1)

        logits = self.mlp(all_actions).squeeze(2)
        return logits


class FJSPActor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, hidden_layers: int = 2) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=2 * embed_dim,
            output_dim=1,
            num_neurons=[hidden_dim] * hidden_layers,
            hidden_act="Tanh",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )
        self.dummy = nn.Parameter(torch.rand(2 * embed_dim))

    def forward(self, td, ma_emb, ops_emb):
        bs, n_ma = ma_emb.shape[:2]
        # (bs, n_jobs, emb)
        job_emb = gather_by_index(ops_emb, td["next_op"], squeeze=False)

        # (bs, n_jobs, n_ma, emb)
        job_emb_expanded = job_emb.unsqueeze(2).expand(-1, -1, n_ma, -1)
        ma_emb_expanded = ma_emb.unsqueeze(1).expand_as(job_emb_expanded)

        # Input of actor MLP
        # shape: [bs, num_jobs * n_ma, 2*emb]
        h_actions = torch.cat((job_emb_expanded, ma_emb_expanded), dim=-1).flatten(1, 2)
        no_ops = self.dummy[None, None].expand(bs, 1, -1)  # [bs, 1, 2*emb_dim]
        # [bs, num_jobs * n_ma + 1, 2*emb_dim]
        h_actions_w_noop = torch.cat((no_ops, h_actions), 1)
        # (b, j*m)
        logits = self.mlp(h_actions_w_noop).squeeze(-1)
        return logits


class L2DDecoder(AutoregressiveDecoder):
    # feature extractor + actor
    def __init__(
        self,
        env_name: str = "jssp",
        feature_extractor: nn.Module = None,
        actor: nn.Module = None,
        init_embedding: nn.Module = None,
        embed_dim: int = 128,
        actor_hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
    ):
        super(L2DDecoder, self).__init__()

        if feature_extractor is None:
            feature_extractor = HetGNNEncoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
                stepwise=True,
            )

        self.feature_extractor = feature_extractor

        if actor is None:
            if env_name == "fjsp":
                actor = FJSPActor(
                    embed_dim=embed_dim, hidden_dim=actor_hidden_dim, hidden_layers=2
                )
            else:
                actor = JSSPActor(
                    embed_dim=embed_dim, hidden_dim=actor_hidden_dim, hidden_layers=2
                )

        self.actor = actor

    def forward(self, td, hidden, num_starts):
        if hidden is None:
            # (bs, n_m, e), (bs, n_j * n_ops, e)
            hidden, _ = self.feature_extractor(td)

        if num_starts > 1:
            hidden = tuple(map(lambda x: batchify(x, num_starts), hidden))

        ma_emb, op_emb = hidden
        # (bs, n_j, e)
        logits = self.actor(td, ma_emb, op_emb)
        # (b, 1 + j)
        mask = td["action_mask"]

        return logits, mask
