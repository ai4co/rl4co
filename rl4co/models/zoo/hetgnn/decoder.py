import torch
import torch.nn as nn

from rl4co.models.common.constructive.autoregressive import AutoregressiveDecoder
from rl4co.models.nn.mlp import MLP
from rl4co.utils.ops import batchify, gather_by_index


class HetGNNDecoder(AutoregressiveDecoder):
    def __init__(
        self, embed_dim, feed_forward_hidden_dim: int = 64, feed_forward_layers: int = 2
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=2 * embed_dim,
            output_dim=1,
            num_neurons=[feed_forward_hidden_dim] * feed_forward_layers,
        )
        self.dummy = nn.Parameter(torch.rand(2 * embed_dim))

    def pre_decoder_hook(self, td, env, hidden, num_starts):
        return td, env, hidden

    def forward(self, td, hidden, num_starts):
        if num_starts > 1:
            hidden = tuple(map(lambda x: batchify(x, num_starts), hidden))

        ma_emb, ops_emb = hidden
        bs, n_rows, emb_dim = ma_emb.shape

        # (bs, n_jobs, emb)
        job_emb = gather_by_index(ops_emb, td["next_op"])

        # (bs, n_jobs, n_ma, emb)
        job_emb_expanded = job_emb.unsqueeze(2).expand(-1, -1, n_rows, -1)
        ma_emb_expanded = ma_emb.unsqueeze(1).expand_as(job_emb_expanded)

        # Input of actor MLP
        # shape: [bs, num_jobs * n_ma, 2*emb]
        h_actions = torch.cat((job_emb_expanded, ma_emb_expanded), dim=-1).flatten(1, 2)
        no_ops = self.dummy[None, None].expand(bs, 1, -1)  # [bs, 1, 2*emb_dim]
        # [bs, num_jobs * n_ma + 1, 2*emb_dim]
        h_actions_w_noop = torch.cat((no_ops, h_actions), 1)

        # (b, j*m)
        mask = td["action_mask"]

        # (b, j*m)
        logits = self.mlp(h_actions_w_noop).squeeze(-1)

        return logits, mask
