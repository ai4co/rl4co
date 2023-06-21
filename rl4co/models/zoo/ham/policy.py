import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.am.decoder import Decoder
from rl4co.models.zoo.ham.encoder import GraphHeterogeneousAttentionEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class HeterogeneousAttentionModelPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        mask_inner: bool = True,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(HeterogeneousAttentionModelPolicy, self).__init__()
        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        self.env = env

        self.encoder = (
            GraphHeterogeneousAttentionEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                env=self.env,
                normalization=normalization,
            )
            if encoder is None
            else encoder
        )

        self.decoder = (
            Decoder(
                self.env,
                embedding_dim,
                num_heads,
                mask_inner=mask_inner,
                force_flash_attn=force_flash_attn,
            )
            if decoder is None
            else decoder
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
        # Encode inputs
        embeddings, _ = self.encoder(td)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # Main rollout: autoregressive decoding
        log_p, actions, td_out = self.decoder(td, embeddings, **decoder_kwargs)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(log_p, actions, td_out.get("mask", None))
        out = {
            "reward": td_out["reward"],
            "log_likelihood": ll,
        }
        if return_actions:
            out["actions"] = actions

        return out
