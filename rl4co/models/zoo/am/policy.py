from typing import Callable, Union

import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder


class AttentionModelPolicy(AutoregressivePolicy):
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
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kwargs,  # TODO
    ):
        if encoder is None:
            encoder = AttentionModelEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                env_name=env_name,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                init_embedding=init_embedding,
                sdpa_fn=sdpa_fn,
            )

        if decoder is None:
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
            )

        super(AttentionModelPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
