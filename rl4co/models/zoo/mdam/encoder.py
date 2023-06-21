import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        input_dim,
        embed_dim=None,
        val_dim=None,
        key_dim=None,
        last_one=False,
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(num_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(num_heads, key_dim, embed_dim))

        self.init_parameters()
        self.last_one = last_one

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
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.num_heads, batch_size, graph_size, -1)
        shp_q = (self.num_heads, batch_size, n_query, -1)

        # Calculate queries, (num_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (num_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (num_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.num_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim),
        ).view(batch_size, n_query, self.embed_dim)
        if self.last_one:
            return (out, attn, V)
        return out


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()

        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_layers,
        feed_forward_hidden=512,
        normalization="batch",
    ):
        args_tuple = []
        for _ in range(num_layers):
            args_tuple += [
                SkipConnection(
                    MultiHeadAttention(
                        num_heads, input_dim=embed_dim, embed_dim=embed_dim
                    )
                ),
                Normalization(embed_dim, normalization),
                SkipConnection(
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(),
                        nn.Linear(feed_forward_hidden, embed_dim),
                    )
                    if feed_forward_hidden > 0
                    else nn.Linear(embed_dim, embed_dim)
                ),
                Normalization(embed_dim, normalization),
            ]
        args_tuple = tuple(args_tuple)

        super(MultiHeadAttentionLayer, self).__init__(*args_tuple)


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        num_layers,
        node_dim=None,
        normalization="batch",
        feed_forward_hidden=512,
        use_native_sdpa=False,  # TODO
        force_flash_attn=False,
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = MultiHeadAttentionLayer(
            num_heads, embed_dim, num_layers - 1, feed_forward_hidden, normalization
        )
        self.attention_layer = MultiHeadAttention(
            num_heads, input_dim=embed_dim, embed_dim=embed_dim, last_one=True
        )
        self.BN1 = Normalization(embed_dim, normalization)
        self.projection = SkipConnection(
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )
        self.BN2 = Normalization(embed_dim, normalization)

    def forward(self, x, mask=None, return_transform_loss=False):
        """
        Returns:
            - h [batch_size, graph_size, embed_dim]
            - attn [num_head, batch_size, graph_size, graph_size]
            - V [num_head, batch_size, graph_size, key_dim]
            - h_old [batch_size, graph_size, embed_dim]
        """
        assert mask is None, "TODO mask not yet supported!"

        h_embeded = x
        h_old = self.layers(h_embeded)
        h_new, attn, V = self.attention_layer(h_old)
        h = h_new + h_old
        h = self.BN1(h)
        h = self.projection(h)
        h = self.BN2(h)

        return (h, h.mean(dim=1), attn, V, h_old)

    def change(self, attn, V, h_old, mask, is_tsp=False):
        num_heads, batch_size, graph_size, feat_size = V.size()
        attn = (1 - mask.float()).view(1, batch_size, 1, graph_size).repeat(
            num_heads, 1, graph_size, 1
        ) * attn
        if is_tsp:
            attn = attn / (
                torch.sum(attn, dim=-1).view(num_heads, batch_size, graph_size, 1)
            )
        else:
            attn = attn / (
                torch.sum(attn, dim=-1).view(num_heads, batch_size, graph_size, 1) + 1e-9
            )
        heads = torch.matmul(attn, V)

        h_new = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.attention_layer.num_heads * self.attention_layer.val_dim),
            self.attention_layer.W_out.view(-1, self.attention_layer.embed_dim),
        ).view(batch_size, graph_size, self.attention_layer.embed_dim)
        h = h_new + h_old
        h = self.BN1(h)
        h = self.projection(h)
        h = self.BN2(h)

        return (h, h.mean(dim=1))
