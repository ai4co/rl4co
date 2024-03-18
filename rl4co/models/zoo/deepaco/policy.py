from typing import Optional, Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.common.nonautoregressive.policy import NonAutoregressivePolicy
from rl4co.models.zoo.deepaco.decoder import DeepACODecoder


class DeepACOPolicy(NonAutoregressivePolicy):
    """Implememts DeepACO policy based on :class:`NonAutoregressivePolicy`.

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        heatmap_generator: Model to use for converting the edge embeddings to the heuristic information.
            If None, use the default MLP defined in :class:`~rl4co.models.zoo.common.nonautoregressive.decoder.EdgeHeatmapGenerator`.
        embedding_dim: Dimension of the embeddings
        num_encoder_layers: Number of layers in the encoder
        num_decoder_layers: Number of layers in the decoder
        **decoder_kwargs: Additional arguments to be passed to the DeepACO decoder.
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        encoder: Optional[nn.Module] = None,
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        embedding_dim: int = 64,
        num_encoder_layers: int = 15,
        num_decoder_layers: int = 5,
        **decoder_kwargs,
    ):
        env_name_: str = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        decoder = DeepACODecoder(
            env_name=env_name_,
            embedding_dim=embedding_dim,
            num_layers=num_decoder_layers,
            heatmap_generator=heatmap_generator,
            **decoder_kwargs,
        )

        super(DeepACOPolicy, self).__init__(
            env_name,
            encoder,
            decoder,
            init_embedding,
            edge_embedding,
            embedding_dim,
            num_encoder_layers,
            num_decoder_layers,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )
