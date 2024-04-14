from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.models.nn.env_embeddings.context import FFSPContext
from rl4co.models.zoo.common.autoregressive.decoder import AutoregressiveDecoder
from rl4co.utils.decoding import process_logits


@dataclass
class PrecomputedCache:
    node_embeddings: Union[Tensor, TensorDict]
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
            logit_key=logit_key,
        )


class MatNetFFSPDecoder(AutoregressiveDecoder):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        use_graph_context: bool = False,
        **kwargs,
    ):
        context_embedding = FFSPContext(embedding_dim)

        super().__init__(
            env_name="ffsp",
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            **kwargs,
        )

        self.no_job_emb = nn.Parameter(
            torch.rand(1, 1, embedding_dim), requires_grad=True
        )

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
            batch_size=bs,
        )
        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
        )


class MultiStageFFSPDecoder(MatNetFFSPDecoder):
    """Decoder class for the solving the FFSP using a seperate MatNet decoder for each stage
    as originally implemented by Kwon et al. (2021)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        use_graph_context: bool = True,
        **kwargs,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            use_graph_context=use_graph_context,
            **kwargs,
        )
        self.cached_embs: PrecomputedCache = None
        # self.encoded_wait_op = nn.Parameter(torch.rand((1, 1, embedding_dim)))

    def _precompute_cache(self, embeddings: Tuple[Tensor], td: TensorDict = None):
        self.cached_embs = super()._precompute_cache(embeddings, td)

    def forward(
        self,
        td: TensorDict,
        decode_type="sampling",
        num_starts: int = 1,
        **decoding_kwargs,
    ) -> Tuple[Tensor, Tensor, TensorDict]:
        device = td.device
        batch_size = td.size(0)

        logits, mask = self._get_logits(self.cached_embs, td, num_starts)
        logprobs = process_logits(
            logits,
            mask,
            **decoding_kwargs,
        )
        all_job_probs = logprobs.exp()

        if "sampling" in decode_type:
            # to fix pytorch.multinomial bug on selecting 0 probability elements
            while True:
                job_selected = all_job_probs.multinomial(1).squeeze(dim=1)
                # shape: (batch)
                job_prob = all_job_probs.gather(1, job_selected[:, None]).squeeze(dim=1)
                # shape: (batch)
                assert (job_prob[td["done"].squeeze()] == 1).all()

                if (job_prob != 0).all():
                    break

        elif "greedy" in decode_type:
            job_selected = all_job_probs.argmax(dim=1)
            # shape: (batch)
            job_prob = torch.zeros(
                size=(batch_size,), device=device
            )  # any number is okay

        else:
            raise ValueError(f"decode type {decode_type} not understood")

        return job_selected, job_prob
