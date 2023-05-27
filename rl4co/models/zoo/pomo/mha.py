import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, feed_forward_hidden: int):
        super().__init__()
        # self.model_params = model_params

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.feed_forward_hidden = feed_forward_hidden
        self.qkv_dim = qkv_dim = embedding_dim // num_heads

        self.Wq = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, num_heads * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(num_heads * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(embedding_dim)
        self.feedForward = Feed_Forward_Module(embedding_dim, feed_forward_hidden)
        self.addAndNormalization2 = Add_And_Normalization_Module(embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=self.num_heads)
        k = reshape_by_heads(self.Wk(input1), head_num=self.num_heads)
        v = reshape_by_heads(self.Wv(input1), head_num=self.num_heads)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, embedding_dim: int, feed_forward_hidden: int):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, feed_forward_hidden)
        self.W2 = nn.Linear(feed_forward_hidden, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)
        return self.W2(F.relu(self.W1(input1)))


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(
            batch_s, head_num, n, input_s
        )
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(
            batch_s, head_num, n, input_s
        )

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed
