from typing import Callable, Optional, Union

from tensordict import TensorDict
from torch import Tensor, nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork


class CriticNetwork(nn.Module):
    """We make the critic network compatible with any problem by using encoder for any environment
    Refactored from Kool et al. (2019) which only worked for TSP. In our case, we make it
    compatible with any problem by using the environment init embedding. Note that if no environment
    name and no init embedding are provided, the critic network does not transform the input (i.e.
    it should be a tensor of shape (batch_size, embedding_dim)).

    Args:
        env_name: environment name to solve
        encoder: Encoder to use for the critic
        init_embedding: Initial embedding to use for the critic
        embedding_dim: Dimension of the embeddings
        hidden_dim: Hidden dimension for the feed-forward network
        num_layers: Number of layers for the encoder
        num_heads: Number of heads for the attention
        normalization: Normalization to use for the attention
        sdpa_fn: Scaled dot product function to use for the attention
    """

    def __init__(
        self,
        env_name: str = None,
        encoder: nn.Module = None,
        init_embedding: nn.Module = None,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        sdpa_fn: Optional[Callable] = None,
        **unused_kwargs,
    ):
        super(CriticNetwork, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        if env_name is None and init_embedding is None:
            self.init_embedding = nn.Identity()  # No embedding
        else:
            self.init_embedding = (
                env_init_embedding(self.env_name, {"embedding_dim": embedding_dim})
                if init_embedding is None
                else init_embedding
            )

        self.encoder = (
            GraphAttentionNetwork(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                normalization=normalization,
                feed_forward_hidden=hidden_dim,
                sdpa_fn=sdpa_fn,
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
