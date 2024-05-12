import math

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.ops import Normalization
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class mySequential(nn.Sequential):
    def forward(
        self, *inputs: Union[Tuple[torch.Tensor], torch.Tensor]
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MultiHeadCompat(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    # used for init nn.Parameter
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)  #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility_s2n = torch.matmul(Q, K.transpose(2, 3))

        return compatibility_s2n


class Synth_Attention(nn.Module):
    def __init__(self, n_heads: int, input_dim: int) -> None:
        super().__init__()

        hidden_dim = input_dim // n_heads

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, hidden_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(2 * n_heads, 2 * n_heads),
            nn.ReLU(inplace=True),
            nn.Linear(2 * n_heads, n_heads),
        )

        self.W_out = nn.Parameter(torch.Tensor(n_heads, hidden_dim, input_dim))

        self.init_parameters()

    # used for init nn.Parameter
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h should be (batch_size, n_query, input_dim)
        batch_size, n_query, input_dim = h_fea.size()

        hflat = h_fea.contiguous().view(-1, input_dim)

        shp = (self.n_heads, batch_size, n_query, self.hidden_dim)

        # Calculate queries, (n_heads, batch_size, n_query, hidden_dim)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, n_key)
        compatibility = torch.cat((torch.matmul(Q, K.transpose(2, 3)), aux_att_score), 0)

        attn_raw = compatibility.permute(
            1, 2, 3, 0
        )  # (batch_size, n_query, n_key, n_heads)
        attn = self.score_aggr(attn_raw).permute(
            3, 0, 1, 2
        )  # (n_heads, batch_size, n_query, n_key)
        heads = torch.matmul(
            F.softmax(attn, dim=-1), V
        )  # (n_heads, batch_size, n_query, hidden_dim)

        h_wave = torch.mm(
            heads.permute(1, 2, 0, 3)  # (batch_size, n_query, n_heads, hidden_dim)
            .contiguous()
            .view(
                -1, self.n_heads * self.hidden_dim
            ),  # (batch_size * n_query, n_heads * hidden_dim)
            self.W_out.view(-1, self.input_dim),  # (n_heads * hidden_dim, input_dim)
        ).view(batch_size, n_query, self.input_dim)

        return h_wave, aux_att_score


class SynthAttNormSubLayer(nn.Module):
    def __init__(self, n_heads: int, input_dim: int, normalization: str) -> None:
        super().__init__()

        self.SynthAtt = Synth_Attention(n_heads, input_dim)

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention and Residual connection
        h_wave, aux_att_score = self.SynthAtt(h_fea, aux_att_score)

        # Normalization
        return self.Norm(h_wave + h_fea), aux_att_score


class FFNormSubLayer(nn.Module):
    def __init__(
        self, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(input_dim, feed_forward_hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, input_dim, bias=False),
            )
            if feed_forward_hidden > 0
            else nn.Linear(input_dim, input_dim, bias=False)
        )

        self.Norm = Normalization(input_dim, normalization)

    __call__: Callable[..., torch.Tensor]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class N2SEncoderLayer(nn.Module):
    def __init__(
        self, n_heads: int, input_dim: int, feed_forward_hidden: int, normalization: str
    ) -> None:
        super().__init__()

        self.SynthAttNorm_sublayer = SynthAttNormSubLayer(
            n_heads, input_dim, normalization
        )

        self.FFNorm_sublayer = FFNormSubLayer(
            input_dim, feed_forward_hidden, normalization
        )

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self, h_fea: torch.Tensor, aux_att_score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_wave, aux_att_score = self.SynthAttNorm_sublayer(h_fea, aux_att_score)
        return self.FFNorm_sublayer(h_wave), aux_att_score
