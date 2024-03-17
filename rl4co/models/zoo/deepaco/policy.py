from typing import Optional, Union

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.common.nonautoregressive.policy import NonAutoregressivePolicy
from rl4co.models.zoo.deepaco.decoder import DeepACODecoder


class DeepACOPolicy(NonAutoregressivePolicy):
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
        n_ants: int = 20,
        n_iterations: int = 50,
        **unused_kw,
    ):
        env_name_: str = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        decoder = DeepACODecoder(
            env_name=env_name_,
            embedding_dim=embedding_dim,
            num_layers=num_decoder_layers,
            heatmap_generator=heatmap_generator,
            n_ants=n_ants,
            n_iterations=n_iterations,
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
