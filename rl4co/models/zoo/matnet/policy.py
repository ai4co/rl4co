from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.matnet.encoder import MatNetEncoder
from rl4co.models.zoo.matnet.decoder import MatNetDecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MatNetPolicy(AutoregressivePolicy):
    """MatNet Policy from Kwon et al., 2021.
    Reference: https://arxiv.org/abs/2106.11113

    Warning:
        This implementation is under development and subject to change.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`

    Default paarameters are adopted from the original implementation.
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int = 256,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        normalization: str = "instance",
        init_embedding_kwargs: dict = {"mode": "RandomOneHot"},
        use_graph_context: bool = False,
        **kwargs,
    ):
        if env_name not in ["atsp"]:
            log.error(f"env_name {env_name} is not originally implemented in MatNet")

        super(MatNetPolicy, self).__init__(
            env_name=env_name,
            encoder=MatNetEncoder(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding_kwargs=init_embedding_kwargs,
            ),
            decoder=MatNetDecoder(
                env_name=env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                use_graph_context=use_graph_context,
            ),
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
