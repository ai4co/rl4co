from typing import Union

import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.dec_strategies import DecodingStrategy, get_decoding_strategy
from rl4co.models.nn.graph.graphCNN import GraphCNN
from rl4co.models.nn.mlp import MLP


class Decoder(nn.Module):
    # TODO GraphCNN + actor
    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
    ):
        super(Decoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.graph_cnn = GraphCNN(
            env_name=env_name,
            embedding_dim=embedding_dim,
            num_layers=num_encoder_layers,
        )

        self.actor = MLP(
            input_dim=embedding_dim,
            output_dim=1,
            num_neurons=[embedding_dim] * num_decoder_layers,
            hidden_act="Tanh",
            out_act="Identity",
            input_norm="None",
            output_norm="None",
        )

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
            embeddings, _ = self.graph_cnn(td)
            cand_emb = embeddings.gather(
                1, td["next_op"][..., None].expand(-1, -1, embeddings.size(-1))
            )
            logits = self.actor(cand_emb).squeeze(2)
            mask = ~td["action_mask"]
            logits[mask] = -torch.inf
            log_p = torch.log_softmax(logits, dim=1)
            td = decode_strategy.step(log_p, mask, td)
            td = env.step(td)["next"]

        outputs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        return outputs, actions, td

    def evaluate_action(self, td, action, env):
        embeddings, _ = self.graph_cnn(td)
        logits = self.actor(embeddings)
        mask = ~td["action_mask"]
        logits[mask] = -torch.inf
        log_p = torch.log_softmax(logits, dim=1)
        return log_p
