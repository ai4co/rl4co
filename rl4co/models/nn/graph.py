import torch.nn as nn

from rl4co.models.nn.attention import NativeFlashMHA
from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        feed_forward_hidden=512,
        normalization="batch",
        force_flash_attn=False,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                NativeFlashMHA(embed_dim, num_heads, force_flash_attn=force_flash_attn)
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embedding_dim,
        num_layers,
        env=None,
        normalization="batch",
        feed_forward_hidden=512,
        force_flash_attn=False,
        disable_init_embedding=False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        if not disable_init_embedding:
            self.init_embedding = env_init_embedding(
                env, {"embedding_dim": embedding_dim}
            )
        else:
            log.warning("Disabling init embedding manually for GraphAttentionEncoder")
            self.init_embedding = nn.Identity() # do nothing

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    embedding_dim,
                    feed_forward_hidden,
                    normalization,
                    force_flash_attn,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x, mask=None):
        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds)
        return embeds, init_embeds


import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MessagePassingEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        env=None,
        aggregation="add",
        disable_init_embedding=False,
    ):
        """
        Args:
            aggregation: "add" or "mean", pytorch geometric variable
        """
        super(MessagePassingEncoder, self).__init__(aggr=aggregation)

        # To map input to embedding space
        if not disable_init_embedding:
            self.init_embedding = env_init_embedding(
                env, {"embedding_dim": embedding_dim}
            )
        else:
            log.warning("Disabling init embedding manually for MessagePassingEncoder")
            self.init_embedding = nn.Identity() # do nothing

        self.bias = Parameter(torch.Tensor(embedding_dim))
        self.bias.data.zero_()

    def forward(self, x, edge_index, mask=None):
        # TODO if we set the graph to be fully connected, 
        # then the edge index can be moved to the init function

        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)
        # layers  (batch_size, graph_size, embed_dim)
        edge_index, _ = add_self_loops(edge_index, num_nodes=init_embeds.size(-2))
        # graph normalization
        row, col = edge_index
        deg = degree(col, init_embeds.size(-2), dtype=init_embeds.dtype) # TODO fix the batch processing
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # propagate
        out = self.propagate(edge_index, x=init_embeds, norm=norm)
        # add bias
        out = out + self.bias
        return out, init_embeds

    def message(self, x_j, norm):
        # normalize the node feature
        return norm.view(-1, 1) * x_j


from torch_geometric.nn import GCNConv



class GCNLayer(nn.Sequential):
    def __init__(
        self,
        embed_dim,
    ):
        """
        TODO: split the embed dim and feed forward hidden dim
        """
        super(GCNLayer, self).__init__(
            nn.Sequential(
                nn.GCNConv(embed_dim, embed_dim),
                nn.Tanh(),
            )
        )


class GCNEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        env=None,
        num_layers=2,
        disable_init_embedding=False,
    ):
        super(GCNEncoder, self).__init__()

        # To map input to embedding space
        if not disable_init_embedding:
            self.init_embedding = env_init_embedding(
                env, {"embedding_dim": embedding_dim}
            )
        else:
            log.warning("Disabling init embedding manually for MessagePassingEncoder")
            self.init_embedding = nn.Identity() # do nothing

        self.layers = nn.Sequential(
            *(
                GCNLayer(embedding_dim)
                for _ in range(num_layers)
            )
        )

    def forward(self, x, edge_index, mask=None):
        # TODO if we set the graph to be fully connected, 
        # then the edge index can be moved to the init function

        assert mask is None, "Mask not yet supported!"
        # initial Embedding from features
        init_embeds = self.init_embedding(x)
        # layers  (batch_size, graph_size, embed_dim)
        embeds = self.layers(init_embeds, edge_index)
        return embeds, init_embeds
