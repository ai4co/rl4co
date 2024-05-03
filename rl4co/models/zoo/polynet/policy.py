from typing import Callable, Union


from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.polynet.decoder import PolyNetDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.zoo.matnet.encoder import MatNetEncoder



class PolyNetPolicy(AutoregressivePolicy):
    """
    # TODO
    Attention Model Policy based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    We re-declare the most important arguments here for convenience as in the paper.
    See `AutoregressivePolicy` superclass for more details.

    Args:
        env_name: Name of the environment used to initialize embeddings
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        **kwargs: keyword arguments passed to the `AutoregressivePolicy`
    """

    def __init__(
        self,
        k: int,
        encoder_type: str,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
        **kwargs,  # TODO
    ):
        if encoder_type == "AM":
            encoder = AttentionModelEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                **kwargs
            )
        elif encoder_type == "MatNet":
            kwargs_with_defaults = {"init_embedding_kwargs": {"mode": "RandomOneHot"}}
            kwargs_with_defaults.update(kwargs)
            encoder = MatNetEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                **kwargs_with_defaults
            )

        decoder = PolyNetDecoder(
            k=k,
            encoder_type=encoder_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            **kwargs
        )

        super(PolyNetPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **kwargs
        )
