import torch
import torch.nn as nn

from torchrl.envs import EnvBase
from tensordict.tensordict import TensorDict

from rl4co.models.zoo.am.embeddings import env_init_embedding
from rl4co.models.zoo.am.encoder import GraphAttentionEncoder
from rl4co.models.zoo.am.utils import get_log_likelihood
from rl4co.models.zoo.pomo.decoder import Decoder


class POMOPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        embedding_dim: int,
        hidden_dim: int,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        num_pomo: int = 10,
        num_encode_layers: int = 3,
        normalization: str = "batch",
        num_heads: int = 8,
        checkpoint_encoder: bool = False,
        mask_inner: bool = True,
        force_flash_attn: bool = False,
        **kwargs
    ):
        super(POMOPolicy, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_encode_layers = num_encode_layers
        self.env = env

        self.num_heads = num_heads
        self.checkpoint_encoder = checkpoint_encoder

        self.init_embedding = env_init_embedding(
            self.env.name, {"embedding_dim": embedding_dim}
        )

        self.encoder = (
            GraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embedding_dim,
                num_layers=self.num_encode_layers,
                normalization=normalization,
                force_flash_attn=force_flash_attn,
            )
            if encoder is None
            else encoder
        )

        self.decoder = (
            Decoder(
                env,
                embedding_dim,
                num_heads,
                num_pomo=num_pomo,
                mask_inner=mask_inner,
                force_flash_attn=force_flash_attn,
            )
            if decoder is None
            else decoder
        )
        self.num_pomo = num_pomo

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        decode_type: str = "sampling",
        return_actions: bool = False,
    ) -> TensorDict:
        """Given observation, precompute embeddings and rollout"""

        # Set decoding type for policy, can be also greedy
        embedding = self.init_embedding(td)
        encoded_inputs, _ = self.encoder(embedding)

        # Main rollout
        _log_p, actions, td = self.decoder(td, encoded_inputs, decode_type)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(_log_p, actions, td.get("mask", None))
        out = {
            "reward": td["reward"],
            "log_likelihood": ll,
            "actions": actions if return_actions else None,
        }

        return out
