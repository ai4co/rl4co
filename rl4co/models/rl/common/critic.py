from typing import Union

from tensordict import TensorDict
from torch import Tensor, nn

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork


class CriticNetwork(nn.Module):
    """We make the critic network compatible with any problem by using encoder for any environment
    Refactored from Kool et al. (2019) which only worked for TSP. In our case, we make it
    compatible with any problem by using the environment init embedding.

    Args:
        env_name: environment name to solve
        encoder: Encoder to use for the critic
        embedding_dim: Dimension of the embeddings
        hidden_dim: Hidden dimension for the feed-forward network
        num_layers: Number of layers for the encoder
        num_heads: Number of heads for the attention
        normalization: Normalization to use for the attention
        force_flash_attn: Whether to force the use of flash attention. If True, cast to fp16
    """

    def __init__(
        self,
        env_name: str = None,
        encoder: nn.Module = None,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        force_flash_attn: bool = False,
        **unused_kwargs,
    ):
        super(CriticNetwork, self).__init__()

        if env_name is None:
            self.init_embedding = nn.Identity()
        else:
            self.init_embedding = env_init_embedding(
                env_name, {"embedding_dim": embedding_dim}
            )

        self.encoder = (
            GraphAttentionNetwork(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                normalization=normalization,
                feed_forward_hidden=hidden_dim,
                force_flash_attn=force_flash_attn,
            )
            if encoder is None
            else encoder
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: Union[Tensor, TensorDict]) -> Tensor:
        """Forward pass of the critic network: encode the imput in embedding space and return the value

        Args:
            x: Input containing the environment state. Can be a Tensor or a TensorDict

        Returns:
            Value of the input state
        """

        # Initial embedding of x. This is the identity function if env_name is None.
        x = self.init_embedding(x)
        x = self.encoder(x)
        return self.value_head(x).mean(1)
