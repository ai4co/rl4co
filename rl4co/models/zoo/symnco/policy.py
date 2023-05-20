import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.modules.models import MLP

from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.nn.graph import GraphAttentionEncoder
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.symnco.decoder import Decoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SymNCOPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        projection_head: nn.Module = None,
        num_starts: int = 10,
        num_encoder_layers: int = 3,
        normalization: str = "batch",
        num_heads: int = 8,
        use_graph_context: bool = True,
        mask_inner: bool = True,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(SymNCOPolicy, self).__init__()
        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        print("policy num_starts", num_starts)

        self.env = env

        self.encoder = (
            GraphAttentionEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                env_name=self.env.name,
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
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        return_actions: bool = False,
        **decoder_kwargs,
    ) -> TensorDict:
        """Given observation, precompute embeddings and rollout"""

        # Set decoding type for policy, can be also greedy
        embeddings, init_embeds = self.encoder(td)

        # Main rollout
        log_p, actions, td = self.decoder(td, embeddings, **decoder_kwargs)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(log_p, actions, td.get("mask", None))
        out = {
            "reward": td["reward"],
            "log_likelihood": ll,
            "proj_embeddings": self.projection_head(init_embeds),
        }
        if return_actions:
            out["actions"] = actions

        return out
