import torch
import torch.nn as nn

from torchrl.envs import EnvBase
from tensordict.tensordict import TensorDict

from ncobench.models.co.am.embeddings import env_init_embedding
from ncobench.models.co.am.encoder import GraphAttentionEncoder
from ncobench.models.co.am.decoder import Decoder
from ncobench.models.co.am.utils import get_log_likelihood


class AttentionModelPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_encode_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        checkpoint_encoder: bool = False,
        mask_inner: bool = True,
        force_flash_attn: bool = False,
        **kwargs
    ):
        super(AttentionModelPolicy, self).__init__()

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
                mask_inner=mask_inner,
                force_flash_attn=force_flash_attn,
            )
            if decoder is None
            else decoder
        )

    def forward(
        self,
        td: TensorDict,
        phase: str = "train",
        decode_type: str = "sampling",
        return_actions: bool = False,
    ) -> TensorDict:
        # Encode and get embeddings
        embedding = self.init_embedding(td)
        encoded_inputs, _ = self.encoder(embedding)

        # Decode to get log_p, action and new state
        log_p, actions, td = self.decoder(td, encoded_inputs, decode_type)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(log_p, actions, td.get("mask", None))
        out = {
            "reward": td["reward"],
            "log_likelihood": ll,
            "actions": actions if return_actions else None,
        }
        return out


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from ncobench.envs.tsp import TSPEnv
    from ncobench.data.dataset import TorchDictDataset

    env = TSPEnv(num_loc=10).transform()

    init_td = env.reset(batch_size=[10000])
    dataset = TorchDictDataset(init_td)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,  # no need to shuffle, we're resampling every epoch
        num_workers=0,
        collate_fn=torch.stack,  # we need this to stack the batches in the dataset
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AttentionModelPolicy(
        env,
        embedding_dim=128,
        hidden_dim=128,
        num_encode_layers=3,
        # force_flash_attn=True,
    ).to(device)

    x = next(iter(dataloader)).to(device)

    print("Input TensorDict shape: ", x.shape)

    out = model(x, decode_type="sampling")
    print("Out reward shape: ", out["reward"].shape)
