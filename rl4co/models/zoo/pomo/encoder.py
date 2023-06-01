# NOTE: this is a sanity check against the MHA implementation in the original POMO codebase
# The major changes compared to the original AM are:
# 1. encoding layers 3 -> 6
# 2. normalization: batch -> instance
# 3. slightly different MHA implementation
# By default, we will use our implementation that is more similar to the original AM
# and just changes the default parameters

import torch.nn as nn

from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.zoo.pomo.mha import MultiHeadAttentionLayer
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embedding_dim,
        num_layers,
        env=None,
        normalization="batch",  # unused
        feed_forward_hidden=512,
        force_flash_attn=False,  # unused
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
            self.init_embedding = nn.Identity()  # do nothing

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    num_heads,
                    embedding_dim,
                    feed_forward_hidden,
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
