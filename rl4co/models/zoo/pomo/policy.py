from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.utils.ops import select_start_nodes


class POMOPolicy(AutoregressivePolicy):
    """POMO model policy based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011.
    We re-declare the most important arguments here for convenience as in the paper.
    See :class:`AutoregressivePolicy` superclass for more details.

    Note:
        Although the policy is the base :class:`AutoregressivePolicy`, we use the default values used in the paper.
        Differently to the base class:
        - `num_encoder_layers=6` (instead of 3)
        - `normalization="instance"` (instead of "batch")
        - `use_graph_context=False` (instead of True)
        The latter is due to the fact that the paper does not use the graph context in the policy, which seems to be
        helpful in overfitting to the training graph size.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        select_start_nodes_fn: Function to select the start nodes for the environment defaulting to :func:`select_start_nodes`
        **kwargs: keyword arguments passed to the :class:`AutoregressivePolicy`
    """

    def __init__(
        self,
        env_name: str,
        embedding_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        use_graph_context: bool = False,
        select_start_nodes_fn: callable = select_start_nodes,
        **kwargs,
    ):
        super(POMOPolicy, self).__init__(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            use_graph_context=use_graph_context,
            select_start_nodes_fn=select_start_nodes_fn,
            **kwargs,
        )
