import torch.nn as nn

from tensordict import TensorDict
from torchrl.envs import EnvBase

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.zoo.mdam.decoder import Decoder
from rl4co.models.zoo.mdam.encoder import GraphAttentionEncoder


class MDAMPolicy(nn.Module):
    """
    Args:
        env: environment to solve
        encoder: encoder module
        decoder: decoder module
        embedding_dim: embedding dimension/hidden dimension
        num_encode_layers: number of layers in encoder
        num_heads: number of heads in multi-head attention
        num_paths: number of paths to sample (specific feature for MDAM)
        eg_step_gap: number of steps between each path sampling (specific feature for MDAM)
        normalization: normalization type
        mask_inner: whether to mask the inner product in attention
        mask_logits: whether to mask the logits in attention
        tanh_clipping: tanh clipping value
        shrink_size: shrink size for the decoder
        use_native_sdpa: whether to use native sdpa (scaled dot product attention)
        force_flash_attn: whether to force use flash attention
        train_decode_type: decode type for training
        val_decode_type: decode type for validation
        test_decode_type: decode type for testing
    """

    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        num_encode_layers: int = 3,
        num_heads: int = 8,
        num_paths: int = 5,
        eg_step_gap: int = 200,
        normalization: str = "batch",
        mask_inner: bool = True,
        mask_logits: bool = True,
        tanh_clipping: float = 10.0,
        shrink_size=None,
        use_native_sdpa: bool = False,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(MDAMPolicy, self).__init__()
        if len(unused_kw) > 0:
            print(f"Unused kwargs: {unused_kw}")

        self.env = env
        self.init_embedding = env_init_embedding(
            self.env.name, {"embedding_dim": embedding_dim}
        )

        self.encoder = (
            GraphAttentionEncoder(
                num_heads=num_heads,
                embed_dim=embedding_dim,
                num_layers=num_encode_layers,
                normalization=normalization,
                use_native_sdpa=use_native_sdpa,
                force_flash_attn=force_flash_attn,
            )
            if encoder is None
            else encoder
        )

        self.decoder = (
            Decoder(
                env=env,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_paths=num_paths,
                mask_inner=mask_inner,
                mask_logits=mask_logits,
                eg_step_gap=eg_step_gap,
                tanh_clipping=tanh_clipping,
                force_flash_attn=force_flash_attn,
                shrink_size=shrink_size,
                train_decode_type=train_decode_type,
                val_decode_type=val_decode_type,
                test_decode_type=test_decode_type,
            )
            if decoder is None
            else decoder
        )

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        return_actions: bool = False,
        **decoder_kwargs,
    ) -> TensorDict:
        embedding = self.init_embedding(td)
        encoded_inputs, _, attn, V, h_old = self.encoder(embedding)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        reward, log_likelihood, kl_divergence, actions = self.decoder(
            td, encoded_inputs, attn, V, h_old, **decoder_kwargs
        )
        out = {
            "reward": reward,
            "log_likelihood": log_likelihood,
            "kl_divergence": kl_divergence,
            "actions": actions if return_actions else None,
        }
        return out
