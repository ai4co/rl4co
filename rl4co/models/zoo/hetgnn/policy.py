from typing import Optional

import torch.nn as nn

from rl4co.models.common.constructive.autoregressive import (
    AutoregressiveDecoder,
    AutoregressiveEncoder,
    AutoregressivePolicy,
)
from rl4co.utils.pylogger import get_pylogger

from .decoder import HetGNNDecoder
from .encoder import HetGNNEncoder

log = get_pylogger(__name__)


class HetGNNPolicy(AutoregressivePolicy):
    """
    Base Non-autoregressive policy for NCO construction methods.
    This creates a heatmap of NxN for N nodes (i.e., heuristic) that models the probability to go from one node to another for all nodes.

    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (non-autoregressively) to construct the solution to the NCO problem

    Warning:
        The effectiveness of the non-autoregressive approach can vary significantly across different problem types and configurations.
        It may require careful tuning of the model architecture and decoding strategy to achieve competitive results.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        decoder: Decoder module. Note that this moule defaults to the non-autoregressive decoder
        embed_dim: Dimension of the embeddings
        env_name: Name of the environment used to initialize embeddings
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        graph_network: Model to use for the graph network. If None, use the default embedding for the environment
        heatmap_generator: Model to use for the heatmap generator. If None, use the default embedding for the environment
        num_layers_heatmap_generator: Number of layers in the heatmap generator
        num_layers_graph_encoder: Number of layers in the graph encoder
        act_fn: Activation function to use in the encoder
        agg_fn: Aggregation function to use in the encoder
        linear_bias: Whether to use bias in the encoder
        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **constructive_policy_kw: Unused keyword arguments
    """

    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjsp",
        init_embedding: Optional[nn.Module] = None,
        linear_bias: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            encoder = HetGNNEncoder(
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                init_embedding=init_embedding,
                linear_bias=linear_bias,
            )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = HetGNNDecoder(
                embed_dim=embed_dim,
                feed_forward_hidden_dim=embed_dim,
                feed_forward_layers=2,
            )
        else:
            # check if the decoder has trainable parameters
            if any(p.requires_grad for p in decoder.parameters()):
                log.error(
                    "The decoder contains trainable parameters. This should not happen in a non-autoregressive policy."
                )

        # Pass to constructive policy
        super(HetGNNPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )
