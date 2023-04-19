from dataclasses import dataclass

import torch
import torch.nn as nn

from rl4co.models.nn.attention import LogitAttention
from rl4co.models.co.am.context import env_context
from rl4co.models.co.am.embeddings import env_dynamic_embedding
from rl4co.models.co.am.utils import decode_probs


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

    def forward(self, td, embeddings, decode_type="sampling"):
        outputs = []
        actions = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute(embeddings)

        while not td[
            "done"
        ].any():  # NOTE: here we suppose all the batch is done at the same time
            log_p, mask = self._get_log_p(cached_embeds, td)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            action = decode_probs(
                log_p.exp().squeeze(1), mask.squeeze(1), decode_type=decode_type
            )

            td.set("action", action[:, None])
            td = self.env.step(td)["next"]

            # Collect output of step
            outputs.append(log_p.squeeze(1))
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
        ) = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # Organize in a TensorDict for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=self.project_fixed_context(graph_embed)[:, None, :],
            glimpse_key=self.logit_attention._make_heads(glimpse_key_fixed),
            glimpse_val=self.logit_attention._make_heads(glimpse_val_fixed),
            logit_key=logit_key_fixed,
        )

        return cached_embeds

    def _get_log_p(self, cached, td):
        step_context = self.context(cached.node_embeddings, td)  # [batch, 1, embed_dim]
        query = cached.graph_context + step_context  # [batch, 1, embed_dim]

        # Compute keys and values for the nodes
        # glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(cached, td['observation'])
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
        mask = mask.unsqueeze(1) if mask.dim() == 2 else mask

        # Compute logits
        log_p = self.logit_attention(query, glimpse_key, glimpse_key, logit_key, mask)

        return log_p, mask
