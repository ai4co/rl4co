import torch
import torch.nn as nn

from einops import einsum

from rl4co.models.common.constructive.autoregressive import AutoregressiveDecoder
from rl4co.models.nn.env_embeddings.init import FJSPFeatureEmbedding
from rl4co.models.nn.mlp import MLP
from rl4co.models.zoo.hetgnn.encoder import HetGNNEncoder
from rl4co.utils.ops import gather_by_index


class L2DDecoder(AutoregressiveDecoder):
    # feature extractor + actor
    def __init__(
        self,
        env_name: str = "l2d",
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
            init_embedding = FJSPFeatureEmbedding(embed_dim=embed_dim, norm_coef=1000)
            feature_extractor = HetGNNEncoder(
                env_name=env_name,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
            )
            # init_emb = JSSPInitEmbedding(embed_dim=embed_dim, use_pos_enc=True)
            # feature_extractor = MatNetEncoder(
            #     embed_dim=embed_dim,
            #     num_heads=num_heads,
            #     num_layers=num_encoder_layers,
            #     normalization="batch",
            #     init_embedding=init_emb
            # )

        self.feature_extractor = feature_extractor
        self.dummy = nn.Parameter(torch.rand(2 * embed_dim))

        if actor is None:
            actor = MLP(
                input_dim=2 * embed_dim,
                output_dim=1,
                num_neurons=[actor_hidden_dim] * 2,
                hidden_act="Tanh",
                out_act="Identity",
                input_norm="None",
                output_norm="None",
            )

        self.actor = actor

    def forward(self, td, *args, **kwargs):
        bs = td.size(0)
        # (bs, n_j)
        next_op = td["next_op"]
        # (bs, n_m, e), (bs, n_j * n_ops, e)
        (ma_emb, op_emb), _ = self.feature_extractor(td)
        # (bs, n_j, e)
        job_emb = gather_by_index(op_emb, next_op, dim=1)
        ma_emb_per_op = einsum(td["ops_ma_adj"], ma_emb, "b m o, b m e -> b o e")
        ma_emb_per_job = gather_by_index(ma_emb_per_op, next_op, dim=1)
        job_emb = torch.cat((job_emb, ma_emb_per_job), dim=2)

        no_ops = self.dummy[None, None].expand(bs, 1, -1)  # [bs, 1, 2*emb_dim]
        # [bs, 1 + num_j, 2*emb_dim]
        all_actions = torch.cat((no_ops, job_emb), 1)

        logits = self.actor(all_actions).squeeze(2)

        # (b, 1 + j)
        mask = td["action_mask"]

        return logits, mask
