from torch import nn

from rl4co.models.nn.graph import GraphAttentionEncoder
from rl4co.models.nn.env_embedding import env_init_embedding


class CriticNetwork(nn.Module):
    def __init__(
        self,
        env,
        encoder=None,
        embedding_dim=128,
        hidden_dim=512,
        n_layers=3,
        num_heads=8,
        encoder_normalization="batch",
    ):
        """We make the critic network compatible with any problem by using init embeddings
        for any environment (original code only works for TSP)

        Args:
            env (EnvBase): environment
            encoder (nn.Module, optional): encoder. Defaults to None. Initialized with GraphAttentionEncoder.
            embedding_dim (int, optional): embedding dimension. Defaults to 128.
            hidden_dim (int, optional): hidden dimension. Defaults to 512.
            n_layers (int, optional): number of encoder layers. Defaults to 3.
            num_heads (int, optional): number of attention heads. Defaults to 8.
            encoder_normalization (str, optional): normalization. Defaults to "batch".
        """
        super(CriticNetwork, self).__init__()

        self.init_embedding = env_init_embedding(
            env.name, {"embedding_dim": embedding_dim}
        )

        self.encoder = (
            GraphAttentionEncoder(
                n_heads=num_heads,
                embed_dim=embedding_dim,
                n_layers=n_layers,
                normalization=encoder_normalization,
                feed_forward_hidden=hidden_dim,
            )
            if encoder is None
            else encoder
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, td):
        embedding = self.init_embedding(td)
        graph_embeddings = self.encoder(embedding)
        # graph_embedings: [batch_size, graph_size, input_dim]
        return self.value_head(graph_embeddings.mean(1))
