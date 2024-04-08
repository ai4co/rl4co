from typing import Union

import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.dec_strategies import DecodingStrategy, get_decoding_strategy
from rl4co.models.nn.mlp import MLP
from rl4co.models.zoo.common.autoregressive.encoder import GraphAttentionEncoder


class Decoder(nn.Module):
    # feature extractor + actor
    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        feature_extractor: nn.Module = None,
        actor: nn.Module = None,
        init_embedding: nn.Module = None,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
    ):
        super(Decoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        if feature_extractor is None:
            feature_extractor = GraphAttentionEncoder(
                env_name=self.env_name,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                normalization=normalization,
                init_embedding=init_embedding,
            )

        self.feature_extractor = feature_extractor

        if actor is None:
            actor = MLP(
                input_dim=embedding_dim,
                output_dim=1,
                num_neurons=[embedding_dim] * 2,
                hidden_act="Tanh",
                out_act="Identity",
                input_norm="None",
                output_norm="None",
            )
        self.actor = actor

    def forward(
        self, td, env, decode_type="sampling", calc_reward: bool = True, **strategy_kwargs
    ):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """
        # Instantiate environment if needed
        if isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Setup decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type, **strategy_kwargs
        )

        while not td["done"].all():
            embeddings, _ = self.feature_extractor(td)
            logits = self.actor(embeddings).squeeze(2)
            mask = ~td["action_mask"]
            logits[mask] = -torch.inf
            log_p = torch.log_softmax(logits, dim=1)
            td = decode_strategy.step(log_p, mask, td)
            td = env.step(td)["next"]

        outputs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        return outputs, actions, td
