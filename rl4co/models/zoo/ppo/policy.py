from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy


class PPOPolicy(AutoregressivePolicy):
    """PPO Policy. The backbone model is inspired by the Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    This is simply a wrapper around the `AutoregressivePolicy` class. PPO needs an `evaluate_actions` method
    inside `AutoregressivePolicy` to work properly to obtain log probabilities and entropy of actions
    under the current policy.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        **kwargs,
    ):
        super(PPOPolicy, self).__init__(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
