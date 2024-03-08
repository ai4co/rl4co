from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.zoo.common.autoregressive.decoder import AutoregressiveDecoder
from rl4co.utils.ops import select_start_nodes


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class MatNetDecoder(AutoregressiveDecoder):
    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], td: TensorDict = None):
        col_emb, row_emb = embeddings
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            col_emb
        ).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(col_emb.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=row_emb,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            # logit_key=col_emb,
            logit_key=logit_key,
        )


class MatNetFFSPDecoder(AutoregressiveDecoder):
    def __init__(
        self,
        env_name,
        embedding_dim: int,
        num_heads: int,
        use_graph_context: bool = True,
        select_start_nodes_fn: callable = select_start_nodes,
        linear_bias: bool = False,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        **logit_attn_kwargs,
    ):
        super().__init__(
            env_name,
            embedding_dim,
            num_heads,
            use_graph_context,
            select_start_nodes_fn,
            linear_bias,
            context_embedding,
            dynamic_embedding,
            **logit_attn_kwargs,
        )

        self.no_job_emb = nn.Parameter(torch.rand(1, 1, embedding_dim))

    def _precompute_cache(self, embeddings: Tuple[Tensor, Tensor], td: TensorDict = None):
        job_emb, ma_emb = embeddings

        bs, _, emb_dim = job_emb.shape

        job_emb_plus_one = torch.cat(
            (job_emb, self.no_job_emb.expand((bs, 1, emb_dim))), dim=1
        )

        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            job_emb_plus_one
        ).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(job_emb_plus_one.mean(1))
        else:
            graph_context = 0

        embeddings = TensorDict(
            {"job_embeddings": job_emb_plus_one, "machine_embeddings": ma_emb},
            batch_size=job_emb.size(0),
        )
        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            # logit_key=col_emb,
            logit_key=logit_key,
        )
