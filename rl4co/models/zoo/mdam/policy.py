from typing import Union

import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.zoo.common.autoregressive import AutoregressivePolicy
from rl4co.models.zoo.mdam.decoder import Decoder
from rl4co.models.zoo.mdam.encoder import GraphAttentionEncoder
from rl4co.utils.pylogger import get_pylogger
import inspect


log = get_pylogger(__name__)


class MDAMPolicy(AutoregressivePolicy):
    """Multi-Decoder Attention Model (MDAM) policy.
    Args:

    """

    def __init__(
        self,
        env_name: str,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        init_embedding: nn.Module = None,        
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        **kwargs,
    ):
        # get feed_forward_hidden and sdpa_fn from kwargs, if exist, otherwise, use default values        
        sig = inspect.signature(GraphAttentionEncoder)
        feed_forward_hidden_default = sig.parameters['feed_forward_hidden'].default
        sdpa_fn_default = sig.parameters['sdpa_fn'].default
        
        if encoder is None:
            encoder = GraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embedding_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                feed_forward_hidden=kwargs.get("feed_forward_hidden", feed_forward_hidden_default),
                sdpa_fn=kwargs.get("sdpa_fn", sdpa_fn_default),
            )
        else:
            encoder = encoder
        
        if decoder is None:
            decoder = Decoder(
                env_name=env_name,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                **kwargs,
            )
        else:
            decoder = decoder        
        
        super(MDAMPolicy, self).__init__(
            env_name=env_name,            
            encoder=encoder,
            decoder=decoder,
            embedding_dim=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )

        if init_embedding is None:
            init_embedding = env_init_embedding(
                env_name, {"embedding_dim": embedding_dim}
            )
        else:
            init_embedding = init_embedding
            
        self.init_embedding = init_embedding

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
        phase: str = "train",
        return_actions: bool = False,
        **decoder_kwargs,
    ) -> TensorDict:
        embedding = self.init_embedding(td)        
        encoded_inputs, _, attn, V, h_old = self.encoder(embedding)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        reward, log_likelihood, kl_divergence, actions = self.decoder(
            td, encoded_inputs, env, attn, V, h_old, **decoder_kwargs
        )
        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "entropy": kl_divergence,
            "actions": actions if return_actions else None,
        }
        return out
