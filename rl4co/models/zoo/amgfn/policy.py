from typing import Callable, Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy

from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AttentionGFNModelPolicy(AutoregressivePolicy):
    """
    Attention Model Policy based on Kool et al. (2019): https://arxiv.org/abs/1803.08475.
    This model first encodes the input graph using a Graph Attention Network (GAT) (:class:`AttentionModelEncoder`)
    and then decodes the solution using a pointer network (:class:`AttentionModelDecoder`). Cache is used to store the
    embeddings of the nodes to be used by the decoder to save computation.
    See :class:`rl4co.models.common.constructive.autoregressive.policy.AutoregressivePolicy` for more details on the inference process.

    Args:
        encoder: Encoder module, defaults to :class:`AttentionModelEncoder`
        decoder: Decoder module, defaults to :class:`AttentionModelDecoder`
        embed_dim: Dimension of the node embeddings
        num_encoder_layers: Number of layers in the encoder
        num_heads: Number of heads in the attention layers
        normalization: Normalization type in the attention layers
        feedforward_hidden: Dimension of the hidden layer in the feedforward network
        env_name: Name of the environment used to initialize embeddings
        encoder_network: Network to use for the encoder
        init_embedding: Module to use for the initialization of the embeddings
        context_embedding: Module to use for the context embedding
        dynamic_embedding: Module to use for the dynamic embedding
        use_graph_context: Whether to use the graph context
        linear_bias_decoder: Whether to use a bias in the linear layer of the decoder
        sdpa_fn_encoder: Function to use for the scaled dot product attention in the encoder
        sdpa_fn_decoder: Function to use for the scaled dot product attention in the decoder
        sdpa_fn: (deprecated) Function to use for the scaled dot product attention
        mask_inner: Whether to mask the inner product
        out_bias_pointer_attn: Whether to use a bias in the pointer attention
        check_nan: Whether to check for nan values during decoding
        temperature: Temperature for the softmax
        tanh_clipping: Tanh clipping value (see Bello et al., 2016)
        mask_logits: Whether to mask the logits during decoding
        train_decode_type: Type of decoding to use during training
        val_decode_type: Type of decoding to use during validation
        test_decode_type: Type of decoding to use during testing
        moe_kwargs: Keyword arguments for MoE,
            e.g., {"encoder": {"hidden_act": "ReLU", "num_experts": 4, "k": 2, "noisy_gating": True},
                   "decoder": {"light_version": True, ...}}
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
        env_name: str = "tsp",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        moe_kwargs: dict = {"encoder": None, "decoder": None},
        **unused_kwargs,
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
                sdpa_fn=sdpa_fn if sdpa_fn_encoder is None else sdpa_fn_encoder,
                moe_kwargs=moe_kwargs["encoder"],
            )

        if decoder is None:
            decoder = AttentionModelDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                sdpa_fn=sdpa_fn if sdpa_fn_decoder is None else sdpa_fn_decoder,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                moe_kwargs=moe_kwargs["decoder"],
            )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
        
        self.Z = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)
        
        logZ = self.Z(hidden).mean(dim=1)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) during decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))
            
        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
            "log_z": logZ,
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict
