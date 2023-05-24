import torch.nn as nn
from torch_geometric.nn import GCNConv

from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


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
                GCNConv(embed_dim, embed_dim),
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
