from typing import Optional, Union

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.zoo.common.nonautoregressive.decoder import NonAutoregressiveDecoder
from rl4co.models.zoo.common.nonautoregressive.encoder import NonAutoregressiveEncoder
from rl4co.utils.decoding import get_log_likelihood
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class NonAutoregressivePolicy(nn.Module):
    """Base Non-autoregressive policy for NCO construction methods.
    The policy performs the following steps:
        1. Encode the environment initial state into node embeddings
        2. Decode (non-autoregressively) to construct the solution to the NCO problem

    Args:
        env_name: Name of the environment used to initialize embeddings
        encoder: Encoder module. Can be passed by sub-classes
        decoder: Decoder module. Can be passed by sub-classes
        init_embedding: Model to use for the initial embedding. If None, use the default embedding for the environment
        edge_embedding: Model to use for the edge embedding. If None, use the default embedding for the environment
        embedding_dim: Dimension of the embeddings
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
        embedding_dim: int = 64,
        num_encoder_layers: int = 15,
        num_decoder_layers: int = 5,
        train_decode_type: str = "multistart_sampling",
        val_decode_type: str = "multistart_greedy",
        test_decode_type: str = "multistart_greedy",
        **unused_kw,
    ):
        super(NonAutoregressivePolicy, self).__init__()

        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        self.env_name = env_name.name if isinstance(env_name, RL4COEnvBase) else env_name

        if encoder is None:
            log.info("Initializing default NonAutoregressiveEncoder")
            self.encoder = NonAutoregressiveEncoder(
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
            return_init_embeds: Whether to return the initial embeddings
            decoder_kwargs: Keyword arguments for the decoder

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # ENCODER: get embeddings from initial state
        graph, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # DECODER: main rollout with autoregressive decoding
        logprobs, actions, td_out = self.decoder(
            td, graph, env, phase=phase, **decoder_kwargs
        )

        out = {"reward": td_out["reward"]}

        if phase == "train":
            # Log likelihood is calculated within the model
            log_likelihood = get_log_likelihood(
                logprobs, actions, td_out.get("mask", None)
            )  # , return_sum=False).mean(-1)
            out["log_likelihood"] = log_likelihood

        if return_actions:
            out["actions"] = actions

        if return_entropy:
            entropy = -(logprobs.exp() * logprobs).nansum(dim=1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch]
            out["entropy"] = entropy

        if return_init_embeds:
            out["init_embeds"] = init_embeds

        return out
