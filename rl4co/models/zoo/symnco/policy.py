import torch
import torch.nn as nn

from torchrl.envs import EnvBase
from torchrl.modules.models import MLP
from tensordict.tensordict import TensorDict

from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.nn.graph import GraphAttentionEncoder
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.symnco.decoder import Decoder


class SymNCOPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        projection_head: nn.Module = None,
        num_starts: int = 10,
        num_encode_layers: int = 3,
        normalization: str = "batch",
        num_heads: int = 8,
        use_graph_context: bool = True,
        mask_inner: bool = True,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
    ):
        super(SymNCOPolicy, self).__init__()

        self.env = env
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_encode_layers = num_encode_layers
        self.num_heads = num_heads
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

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
                num_starts=num_starts,
                use_graph_context=use_graph_context,
                mask_inner=mask_inner,
                force_flash_attn=force_flash_attn,
            )
            if decoder is None
            else decoder
        )
        self.projection_head = (
            MLP(embedding_dim, embedding_dim, 1, embedding_dim, nn.ReLU)
            if projection_head is None
            else projection_head
        )

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        return_actions: bool = False,
        **decoder_kwargs
    ) -> TensorDict:
        """Given observation, precompute embeddings and rollout"""

        # Set decoding type for policy, can be also greedy
        embeddings = self.init_embedding(td)
        encoded_inputs = self.encoder(embeddings)

        # Main rollout
        log_p, actions, td = self.decoder(td, encoded_inputs, **decoder_kwargs)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(log_p, actions, td.get("mask", None))
        out = {
            "reward": td["reward"],
            "log_likelihood": ll,
            "proj_embeddings": self.projection_head(embeddings),
            "actions": actions if return_actions else None,
        }

        return out
