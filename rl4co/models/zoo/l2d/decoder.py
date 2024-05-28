from typing import Tuple

import torch
import torch.nn as nn

from einops import einsum, rearrange
from tensordict import TensorDict

from rl4co.models.common.constructive.autoregressive import AutoregressiveDecoder
from rl4co.models.nn.attention import PointerAttention
from rl4co.models.nn.env_embeddings.context import SchedulingContext
from rl4co.models.nn.env_embeddings.dynamic import JSSPDynamicEmbedding
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.models.nn.mlp import MLP
from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache
from rl4co.utils.ops import batchify, gather_by_index

from .encoder import GCN4JSSP


class JSSPActor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        het_emb: bool = False,
        check_nan: bool = True,
    ) -> None:
        super().__init__()

        input_dim = (1 + int(het_emb)) * embed_dim
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=1,
            num_neurons=[hidden_dim] * hidden_layers,
            hidden_act="ReLU",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )
        self.het_emb = het_emb
        self.dummy = nn.Parameter(torch.rand(input_dim))
        self.check_nan = check_nan

    def forward(self, td, op_emb, ma_emb=None):
        bs = td.size(0)
        # (bs, n_j)
        next_op = td["next_op"]
        # (bs, n_j, emb)
        job_emb = gather_by_index(op_emb, next_op, dim=1)
        if ma_emb is not None:
            ma_emb_per_op = einsum(td["ops_ma_adj"], ma_emb, "b m o, b m e -> b o e")
            # (bs, n_j, emb)
            ma_emb_per_job = gather_by_index(ma_emb_per_op, next_op, dim=1)
            # (bs, n_j, 2 * emb)
            job_emb = torch.cat((job_emb, ma_emb_per_job), dim=2)
        # (bs, n_j, 2 * emb)
        no_ops = self.dummy[None, None].expand(bs, 1, -1)
        # (bs, 1 + n_j, 2 * emb)
        all_actions = torch.cat((no_ops, job_emb), 1)
        # (bs, 1 + n_j)
        logits = self.mlp(all_actions).squeeze(2)

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits


class FJSPActor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        hidden_layers: int = 2,
        check_nan: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=2 * embed_dim,
            output_dim=1,
            num_neurons=[hidden_dim] * hidden_layers,
            hidden_act="ReLU",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )
        self.dummy = nn.Parameter(torch.rand(2 * embed_dim))
        self.check_nan = check_nan

    def forward(self, td, ops_emb, ma_emb):
        bs, n_ma = ma_emb.shape[:2]
        # (bs, n_jobs, emb)
        job_emb = gather_by_index(ops_emb, td["next_op"], squeeze=False)
        # (bs, n_jobs, n_ma, emb)
        job_emb_expanded = job_emb.unsqueeze(2).expand(-1, -1, n_ma, -1)
        ma_emb_expanded = ma_emb.unsqueeze(1).expand_as(job_emb_expanded)
        # (bs, num_jobs * n_ma, 2*emb)
        h_actions = torch.cat((job_emb_expanded, ma_emb_expanded), dim=-1).flatten(1, 2)
        # (bs, 1, 2*emb_dim)
        no_ops = self.dummy[None, None].expand(bs, 1, -1)
        # (bs, num_jobs * n_ma + 1, 2*emb_dim)
        h_actions_w_noop = torch.cat((no_ops, h_actions), 1)
        # (b, j*m)
        logits = self.mlp(h_actions_w_noop).squeeze(-1)

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

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
        actor_hidden_layers: int = 2,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        het_emb: bool = False,
        stepwise: bool = False,
        scaling_factor: int = 1000,
    ):
        super(L2DDecoder, self).__init__()

        if feature_extractor is None and stepwise:
            if env_name == "fjsp" or (het_emb and env_name == "jssp"):
                feature_extractor = HetGNNEncoder(
                    env_name=env_name,
                    embed_dim=embed_dim,
                    num_layers=num_encoder_layers,
                    normalization=normalization,
                    init_embedding=init_embedding,
                    scaling_factor=scaling_factor,
                )
            else:
                feature_extractor = GCN4JSSP(
                    embed_dim,
                    num_encoder_layers,
                    init_embedding=init_embedding,
                    scaling_factor=scaling_factor,
                )

        self.feature_extractor = feature_extractor

        if actor is None:
            if env_name == "fjsp":
                actor = FJSPActor(
                    embed_dim=embed_dim,
                    hidden_dim=actor_hidden_dim,
                    hidden_layers=actor_hidden_layers,
                )
            else:
                actor = JSSPActor(
                    embed_dim=embed_dim,
                    hidden_dim=actor_hidden_dim,
                    hidden_layers=actor_hidden_layers,
                    het_emb=het_emb,
                )

        self.actor = actor

    def forward(self, td, hidden, num_starts):
        if hidden is None:
            # (bs, n_j * n_ops, e), (bs, n_m, e)
            hidden, _ = self.feature_extractor(td)

        elif num_starts > 1:
            hidden = (hidden,) if isinstance(hidden, torch.Tensor) else hidden
            hidden = tuple(map(lambda x: batchify(x, num_starts), hidden))

        else:
            hidden = (hidden,) if isinstance(hidden, torch.Tensor) else hidden

        # (bs, n_j, e)
        logits = self.actor(td, *hidden)
        # (b, 1 + j)
        mask = td["action_mask"]

        return logits, mask


class L2DAttnPointer(PointerAttention):
    def __init__(
        self,
        env_name: str,
        embed_dim: int,
        num_heads: int,
        out_bias: bool = False,
        check_nan: bool = True,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mask_inner=False,
            out_bias=out_bias,
            check_nan=check_nan,
        )
        self.env_name = env_name

    def forward(self, query, key, value, logit_key, attn_mask=None):
        # bs = query.size(0)
        # (b m j)
        logits = super().forward(query, key, value, logit_key, attn_mask=attn_mask)
        if self.env_name == "jssp":
            # (b j)
            logits = logits.sum(1)
        elif self.env_name == "fjsp":
            no_op_logits = logits[..., 0].sum(1, keepdims=True)
            logits = rearrange(logits[..., 1:], "b m j -> b (j m)")
            logits = torch.cat((no_op_logits, logits), dim=1)

        return logits


class L2DAttnDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "jssp",
        scaling_factor: int = 1000,
    ):
        context_embedding = SchedulingContext(embed_dim, scaling_factor=scaling_factor)
        dynamic_embedding = JSSPDynamicEmbedding(embed_dim, scaling_factor=scaling_factor)
        pointer = L2DAttnPointer(env_name, embed_dim, num_heads, check_nan=False)

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            pointer=pointer,
        )
        self.dummy = nn.Parameter(torch.rand(1, embed_dim))

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict):
        embeddings = cached.node_embeddings
        ma_embs = embeddings["machine_embeddings"]
        return self.context_embedding(ma_embs, td)

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            gather_by_index(cached.glimpse_key, td["next_op"], dim=1),
            gather_by_index(cached.glimpse_val, td["next_op"], dim=1),
            gather_by_index(cached.logit_key, td["next_op"], dim=1),
        )
        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td, cached)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        no_ops = self.dummy.unsqueeze(1).expand(td.size(0), 1, -1).to(logit_k)
        logit_k = torch.cat((no_ops, logit_k), dim=1)

        return glimpse_k, glimpse_v, logit_k

    def _precompute_cache(self, embeddings: Tuple[torch.Tensor, torch.Tensor], **kwargs):
        ops_emb, ma_emb = embeddings

        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            ops_emb
        ).chunk(3, dim=-1)

        embeddings = TensorDict(
            {"op_embeddings": ops_emb, "machine_embeddings": ma_emb},
            batch_size=ops_emb.size(0),
        )
        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=0,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )
