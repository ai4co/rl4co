from dataclasses import dataclass

import torch
import torch.nn as nn

from rl4co.utils.ops import batchify
from rl4co.utils import get_pylogger
from rl4co.models.nn.attention import LogitAttention
from rl4co.models.nn.env_context import env_context
from rl4co.models.nn.env_embedding import env_dynamic_embedding
from rl4co.models.nn.utils import decode_probs
from rl4co.models.zoo.pomo.utils import select_start_nodes


log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: torch.Tensor
    graph_context: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor


class Decoder(nn.Module):
    def __init__(
        self,
        env,
        embedding_dim,
        num_heads,
        num_starts=20,
        use_graph_context=True,
        **logit_attn_kwargs
    ):
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

        # POMO
        self.num_starts = max(num_starts, 1)  # POMO = 1 is just normal REINFORCE
        self.use_graph_context = use_graph_context  # disabling makes it like in POMO

    def forward(self, td, embeddings, decode_type="sampling", softmax_temp=None):
        # Collect outputs
        outputs = []
        actions = []

        if self.num_starts > 1:
            # POMO: first action is decided via select_start_nodes
            action = select_start_nodes(
                batch_size=td.shape[0], num_nodes=self.num_starts, device=td.device
            )

            # # Expand td to batch_size * num_starts
            td = batchify(td, self.num_starts)

            td.set("action", action)
            td = self.env.step(td)["next"]
            log_p = torch.zeros_like(
                td["action_mask"], device=td.device
            )  # first log_p is 0, so p = log_p.exp() = 1

            outputs.append(log_p)
            actions.append(action)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute(embeddings)

        while not td["done"].all():
            # Compute the logits for the next node
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
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # In POMO, no graph context (trick for overfit to single graph size) # [batch, 1, embed_dim]
        graph_context = (
            batchify(
                self.project_fixed_context(embeddings.mean(1)),
                self.num_starts,
            )
            if self.use_graph_context
            else 0
        )

        # Organize in a dataclass for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=batchify(embeddings, self.num_starts),
            graph_context=graph_context,
            glimpse_key=batchify(glimpse_key_fixed, self.num_starts),
            glimpse_val=batchify(glimpse_val_fixed, self.num_starts),
            logit_key=batchify(logit_key_fixed, self.num_starts),
        )
        return cached_embeds

    def _get_log_p(self, cached, td, softmax_temp=None):
        # Compute the query based on the context (computes automatically the first and last node context)
        step_context = self.context(cached.node_embeddings, td)  # [batch, embed_dim]
        query = (cached.graph_context + step_context).unsqueeze(
            1
        )  # [batch, 1, embed_dim]

        # Compute keys and values for the nodes
        (
            glimpse_key_dynamic,
            glimpse_val_dynamic,
            logit_key_dynamic,
        ) = self.dynamic_embedding(td)
        glimpse_key = cached.glimpse_key + glimpse_key_dynamic
        glimpse_key = cached.glimpse_val + glimpse_val_dynamic
        logit_key = cached.logit_key + logit_key_dynamic

        # Get the mask
        mask = ~td["action_mask"]

        # Compute logits
        log_p = self.logit_attention(query, glimpse_key, glimpse_key, logit_key, mask, softmax_temp)

        return log_p, mask
