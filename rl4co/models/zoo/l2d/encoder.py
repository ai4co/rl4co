import torch.nn as nn

from rl4co.models.nn.attention import MultiHeadCrossAttention
from rl4co.models.nn.env_embeddings.init import JSSPInitEmbedding
from rl4co.models.nn.graph.gcn import GCNEncoder
from rl4co.models.nn.ops import Normalization
from rl4co.models.zoo.matnet.encoder import MixedScoresSDPA
from rl4co.utils.ops import adj_to_pyg_edge_index


class GCN4JSSP(GCNEncoder):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        init_embedding=None,
        **init_embedding_kwargs,
    ):
        def edge_idx_fn(td, _):
            return adj_to_pyg_edge_index(td["adjacency"])

        if init_embedding is None:
            init_embedding = JSSPInitEmbedding(embed_dim, **init_embedding_kwargs)

        super().__init__(
            env_name="jssp",
            embed_dim=embed_dim,
            num_layers=num_layers,
            edge_idx_fn=edge_idx_fn,
            init_embedding=init_embedding,
        )


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_scores=1,
        feed_forward_hidden=256,
        normalization="batch",
    ):
        super(EncoderBlock, self).__init__()
        ms = MixedScoresSDPA(num_heads, num_scores=num_scores)
        self.cross_attn_block = MultiHeadCrossAttention(embed_dim, num_heads, sdpa_fn=ms)
        self.F_a = nn.ModuleDict(
            {
                "norm1": Normalization(embed_dim, normalization),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                ),
                "norm2": Normalization(embed_dim, normalization),
            }
        )

    def forward(self, x, dmat=None, mask=None):
        x_out = self.cross_attn_block(x, x, cross_attn_mask=mask, dmat=dmat)
        x_emb_out = self.F_a["norm1"](x + x_out)
        x_emb_out = self.F_a["norm2"](x_emb_out + self.F_a["ffn"](x_emb_out))
        return x_emb_out


class AttnEncoder4JSSP(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_layers=3,
        init_embedding: nn.Module = None,
        **init_embedding_kwargs,
    ):
        super().__init__()

        if init_embedding is None:
            init_embedding = JSSPInitEmbedding(embed_dim, **init_embedding_kwargs)

        self.init_embedding = init_embedding
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads, num_scores=1) for _ in range(num_layers)]
        )

    def forward(self, td):
        init_emb = self.init_embedding(td)
        op_emb = init_emb.clone()
        dmat = td["adjacency"]

        for layer in self.layers:
            op_emb = layer(op_emb, dmat=dmat)

        return op_emb, init_emb
