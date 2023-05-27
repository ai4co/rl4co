import sys

sys.path.append("./")

import math
from typing import List, Tuple, Optional, NamedTuple, Dict, Union, Any
from einops import rearrange, repeat
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning as L

from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp
from tensordict import TensorDict

from rl4co.envs import TSPEnv
from rl4co.utils.lightning import get_lightning_device
from rl4co.data.dataset import TensorDictCollate, TensorDictDataset

# from rl4co.models import AttentionModel, AttentionModelPolicy # in this notebook, we make them from scratch.uncomment this line if you want to use the original ones.
from rl4co.models.nn.env_embedding import env_init_embedding
from rl4co.models.nn.graph import GraphAttentionEncoder
from rl4co.models.zoo.am.decoder import Decoder
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.attention import LogitAttention
from rl4co.models.nn.env_context import env_context
from rl4co.models.nn.env_embedding import env_dynamic_embedding
from rl4co.models.nn.utils import decode_probs
from rl4co.models.rl.reinforce.baselines import (
    RolloutBaseline,
    WarmupBaseline,
    ExponentialBaseline,
)
from rl4co.tasks.rl4co import RL4COLitModule


@dataclass
class PrecomputedCache:
    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class Decoder(nn.Module):
    def __init__(self, env, embedding_dim, num_heads, **logit_attn_kwargs):
        super(Decoder, self).__init__()

        self.env = env
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.context = env_context(self.env.name, {"embedding_dim": embedding_dim})
        self.dynamic_embedding = env_dynamic_embedding(
            self.env.name, {"embedding_dim": embedding_dim}
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # MHA
        self.logit_attention = LogitAttention(
            embedding_dim, num_heads, **logit_attn_kwargs
        )

    def forward(self, td, embeddings, decode_type="sampling", softmax_temp=None):
        outputs = []
        actions = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute(embeddings)

        while not td["done"].all():
            log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            action = decode_probs(log_p.exp(), mask, decode_type=decode_type)

            td.set("action", action)
            td = self.env.step(td)["next"]

            # Collect output of step
            outputs.append(log_p)
            actions.append(action)

        outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
        td.set("reward", self.env.get_reward(td, actions))
        return outputs, actions, td

    def _precompute(self, embeddings):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Organize in a TensorDict for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=self.project_fixed_context(graph_embed),
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

        return cached_embeds

    def _get_log_p(self, cached, td, softmax_temp):
        step_context = self.context(cached.node_embeddings, td)  # [batch, embed_dim]
        glimpse_q = (cached.graph_context + step_context).unsqueeze(
            1
        )  # [batch, 1, embed_dim]

        # Compute keys and values for the nodes
        (
            glimpse_key_dynamic,
            glimpse_val_dynamic,
            logit_key_dynamic,
        ) = self.dynamic_embedding(td)
        glimpse_k = cached.glimpse_key + glimpse_key_dynamic
        glimpse_v = cached.glimpse_val + glimpse_val_dynamic
        logit_k = cached.logit_key + logit_key_dynamic

        # Get the mask
        mask = ~td["action_mask"]

        # Compute log prob: MHA + single-head attention
        log_p = self.logit_attention(
            glimpse_q, glimpse_k, glimpse_v, logit_k, mask, softmax_temp
        )

        return log_p, mask


class AttentionModelPolicy(nn.Module):
    def __init__(
        self,
        env: EnvBase,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embedding_dim: int = 128,
        num_encode_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        mask_inner: bool = True,
        force_flash_attn: bool = False,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(AttentionModelPolicy, self).__init__()
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
        # Encode and get embeddings
        embedding = self.init_embedding(td)
        encoded_inputs = self.encoder(embedding)

        # Get decode type depending on phase
        if decoder_kwargs.get("decode_type", None) is None:
            decoder_kwargs["decode_type"] = getattr(self, f"{phase}_decode_type")

        # Decode to get log_p, action and new state
        log_p, actions, td = self.decoder(td, encoded_inputs, **decoder_kwargs)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = get_log_likelihood(log_p, actions, td.get("mask", None))
        out = {
            "reward": td["reward"],
            "log_likelihood": ll,
            "actions": actions if return_actions else None,
        }
        return out


# Load environment with test data
env = TSPEnv()

dataset = env.dataset(batch_size=[10000])

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,  # no need to shuffle, we're resampling every epoch
    num_workers=0,
    collate_fn=TensorDictCollate(),
)

policy = AttentionModelPolicy(
    env,
).to("cuda")

# model = torch.compile(model)

td = next(iter(dataloader)).to("cuda")
td = env.reset(td)

out = policy(td, decode_type="sampling", return_actions=False)
