from typing import Optional

import torch.nn as nn

from rl4co.models.common.constructive.autoregressive import (
    AutoregressiveDecoder,
    AutoregressiveEncoder,
    AutoregressivePolicy,
)
from rl4co.models.common.constructive.base import NoEncoder
from rl4co.models.nn.graph.hgnn import HetGNNEncoder
from rl4co.utils.pylogger import get_pylogger

from .decoder import L2DDecoder

log = get_pylogger(__name__)


class L2DPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[AutoregressiveEncoder] = None,
        decoder: Optional[AutoregressiveDecoder] = None,
        embed_dim: int = 64,
        num_encoder_layers: int = 2,
        env_name: str = "fjsp",
        init_embedding: Optional[nn.Module] = None,
        stepwise_encoding: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "multistart_sampling",
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warn(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            if stepwise_encoding:
                encoder = NoEncoder()
            else:
                encoder = HetGNNEncoder(
                    env_name=env_name,
                    embed_dim=embed_dim,
                    num_layers=num_encoder_layers,
                    normalization="batch",
                    init_embedding=init_embedding,
                    stepwise=False,
                )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = L2DDecoder(
                env_name=env_name,
                embed_dim=embed_dim,
                actor_hidden_dim=embed_dim,
                num_encoder_layers=num_encoder_layers,
                init_embedding=init_embedding,
            )

        # Pass to constructive policy
        super(L2DPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )
