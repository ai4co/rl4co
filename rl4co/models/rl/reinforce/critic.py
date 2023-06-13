from torch import nn

from rl4co.models.nn.graph.gat import GraphAttentionEncoder


class CriticNetwork(nn.Module):
    """We make the critic network compatible with any problem by using encoder for any environment
    Refactored from Kool et al. (2019) which only worked for TSP
    Reference: https://github.com/wouterkool/attention-learn-to-route

    Args:
        env (EnvBase): environment
        encoder (nn.Module, optional): encoder. Defaults to None. Initialized with GraphAttentionEncoder.
        embedding_dim (int, optional): embedding dimension. Defaults to 128.
        hidden_dim (int, optional): hidden dimension. Defaults to 512.
        n_layers (int, optional): number of encoder layers. Defaults to 3.
        num_heads (int, optional): number of attention heads. Defaults to 8.
        encoder_normalization (str, optional): normalization. Defaults to "batch".
    """

    def __init__(
        self,
        env=None,
        encoder=None,
        embedding_dim=128,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        encoder_normalization="batch",
        use_native_sdpa=False,
        force_flash_attn=False,
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = (
            GraphAttentionEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                env=env,
                normalization=encoder_normalization,
                feed_forward_hidden=hidden_dim,
                use_native_sdpa=use_native_sdpa,
                force_flash_attn=force_flash_attn,
            )
            if encoder is None
            else encoder
        )

        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, td):
        graph_embeddings, _ = self.encoder(td)
        # graph_embedings: [batch_size, graph_size, input_dim]
        # return self.value_head(graph_embeddings.mean(1))

        # L2D style
        return self.value_head(graph_embeddings).mean(1)
