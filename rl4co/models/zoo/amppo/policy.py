
import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.envs import EnvBase

from rl4co.models.nn.graph.gat import GraphAttentionEncoder
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.zoo.amppo.decoder import PPODecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PPOAttentionModelPolicy(nn.Module):
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
        use_native_sdpa: bool = False,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(PPOAttentionModelPolicy, self).__init__()
        if len(unused_kw) > 0:
            log.warn(f"Unused kwargs: {unused_kw}")

        self.env = env

        self.encoder = (
            GraphAttentionEncoder(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                num_layers=num_encoder_layers,
                env=self.env,
                normalization=normalization,
                use_native_sdpa=use_native_sdpa,
                force_flash_attn=force_flash_attn,
            )
            if encoder is None
            else encoder
        )

        self.decoder = (
            PPODecoder(
                env,
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
        return_action: bool = False,
        return_entropy: bool = False,
        given_actions: torch.Tensor = None,
        **decoder_kwargs,
    ) -> dict:
        # Encode inputs
        embeddings, _ = self.encoder(td)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # Main rollout: autoregressive decoding
        log_p, actions, td_out = self.decoder(
            td, embeddings, given_actions=given_actions, **decoder_kwargs
        )

        # Log likelihood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(
            log_p, actions, td_out.get("mask", None), return_sum=False
        )

        out = {
            "reward": td_out["reward"],
            "log_likelihood": ll,  # [batch, decoder steps]
        }

        if given_actions is not None:
            selected_log_p = get_log_likelihood(
                log_p, given_actions, td_out.get("mask", None), return_sum=False
            )
            assert selected_log_p.isfinite().all(), "Log p is not finite"
            out["selected_log_p"] = selected_log_p  # [batch, decoder steps]

        if return_action:
            out["actions"] = actions  # [batch, decoder steps]

        if return_entropy:
            # log_p [batch, decoder steps, num nodes]
            log_p = torch.nan_to_num(log_p, nan=0.0)
            entropy = -(log_p.exp() * log_p).sum(dim=-1)  # [batch, decoder steps]
            entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
            assert entropy.isfinite().all(), "Entropy is not finite"
            out["entropy"] = entropy

        return out
