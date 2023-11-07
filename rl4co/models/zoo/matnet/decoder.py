from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from rl4co.models.zoo.common.autoregressive.decoder import AutoregressiveDecoder
from rl4co.utils.ops import batchify, get_num_starts, select_start_nodes, unbatchify
from tensordict import TensorDict
from torch import Tensor


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class MatNetDecoder(AutoregressiveDecoder):
    def _precompute_cache(
        self, embeddings: Tuple[Tensor, Tensor], num_starts: int = 0, td: TensorDict = None
    ):
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
            graph_context = unbatchify(
                batchify(self.project_fixed_context(col_emb.mean(1)), num_starts),
                num_starts,
            )
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