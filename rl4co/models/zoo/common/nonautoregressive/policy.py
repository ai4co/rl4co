from typing import Callable, Optional, Union

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.zoo.common.nonautoregressive.decoder import NonAutoregressiveDecoder
from rl4co.models.zoo.common.nonautoregressive.encoder import AnisotropicGNNEncoder
from rl4co.utils.ops import select_start_nodes
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class NonAutoregressivePolicy(nn.Module):
    """TODO

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes
        decoder: Decoder module. Can be passed by sub-classes
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        select_start_nodes_fn: Function to select the start nodes for multi-start decoding

        embedding_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_decoder_layers: Number of layers in the decoder

        train_decode_type: Type of decoding during training
        val_decode_type: Type of decoding during validation
        test_decode_type: Type of decoding during testing
        **unused_kw: Unused keyword arguments
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        select_start_nodes_fn: Callable = select_start_nodes,
        embedding_dim: int = 32,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 3,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(NonAutoregressivePolicy, self).__init__()

        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        self.env_name = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        if encoder is None:
            log.info("Initializing default AnisotropicGNNEncoder")
            self.encoder = AnisotropicGNNEncoder(
                env_name=self.env_name,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                init_embedding=init_embedding,
                edge_embedding=edge_embedding,
            )
        else:
            self.encoder = encoder

        if decoder is None:
            log.info("Initializing default NonAutoregressiveDecoder")
            self.decoder = NonAutoregressiveDecoder(
                env_name=self.env_name,
                embedding_dim=embedding_dim,
                num_layers=num_decoder_layers,
                select_start_nodes_fn=select_start_nodes_fn,
            )
        else:
            self.decoder = decoder

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        phase: str = "train",
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        **decoder_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding
            phase: Phase of the algorithm (train, val, test)
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            decoder_kwargs: Keyword arguments for the decoder. See :class:`rl4co.models.zoo.common.autoregressive.decoder.AutoregressiveDecoder`

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # ENCODER: get embeddings from initial state
        data = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # DECODER: main rollout with autoregressive decoding
        log_p, actions, td_out = self.decoder(td, data, env, **decoder_kwargs)

        return data
