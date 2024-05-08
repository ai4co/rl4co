import torch
import torch.nn as nn

from rl4co.envs.scheduling.fjsp.utils import get_flat_action_mask
from rl4co.models.common.constructive.autoregressive import AutoregressiveDecoder
from rl4co.models.nn.mlp import MLP


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
        row_emb, col_emb = hidden
        bs, n_rows, emb_dim = row_emb.shape

        # TODO where to put this FJSP exclusive logic?
        next_op = td["next_op"].unsqueeze(-1).expand((-1, -1, emb_dim))
        # (bs, n_jobs, emb)
        job_emb = col_emb.gather(1, next_op)
        # (bs, n_jobs, n_ma, emb)
        job_emb_expanded = job_emb.unsqueeze(-2).expand(-1, -1, n_rows, -1)
        ma_emb_expanded = row_emb.unsqueeze(-3).expand_as(job_emb_expanded)

        # Input of actor MLP
        # shape: [bs, num_mas, num_jobs, 2*emb]
        h_actions = (
            torch.cat((job_emb_expanded, ma_emb_expanded), dim=-1)
            .transpose(1, 2)
            .flatten(1, 2)
        )
        no_ops = self.dummy[None, None].expand(bs, 1, -1)  # [bs, 1, 2*emb_dim]
        h_actions_w_noop = torch.cat(
            (no_ops, h_actions), 1
        )  # [bs, num_shelves * num_skus + 1, 2*emb_dim]

        # (b, m*j)
        mask = get_flat_action_mask(td)

        # (bs, ma*jobs)
        scores = self.mlp(h_actions_w_noop).squeeze(-1)
        return scores, mask
