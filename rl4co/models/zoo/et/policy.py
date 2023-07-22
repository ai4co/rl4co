from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.utils.pylogger import get_logger

log = get_logger(__name__)


class EquityTransformerPolicy(AutoregressivePolicy):
    """Equity Transformer Policy from Son et al., 2023.
    Reference: https://arxiv.org/abs/2306.02689

    Warning:
        This implementation is under development and subject to change.

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
        if env_name not in ["mtsp", "mpdp"]:
            log.error(f"env_name {env_name} is not originally implemented in ET")

        super(EquityTransformerPolicy, self).__init__(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
