"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from rl4co.envs.scheduling.ffsp import FFSPEnv
from rl4co.models.zoo.matnet.encoder import MatNetMHANetwork
from rl4co.utils.ops import batchify


class FFSPModel(nn.Module):
    def __init__(
        self,
        stage_cnt,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **model_params,
    ):
        super().__init__()
        self.stage_cnt = stage_cnt
        self.stage_models = nn.ModuleList(
            [OneStageModel(**model_params) for _ in range(self.stage_cnt)]
        )

        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def pre_forward(self, td: TensorDict, env: FFSPEnv, num_starts: int):
        for stage_idx in range(self.stage_cnt):
            cost_mat = td["cost_matrix"][:, :, :, stage_idx]
            model = self.stage_models[stage_idx]
            model.pre_forward(cost_mat, num_starts)

        if num_starts > 1:
            # repeat num_start times
            td = batchify(td, num_starts)
            # update machine idx and action mask
            td = env.pre_step(td)

        return td

    def soft_reset(self):
        # Nothing to reset
        pass

    def forward(
        self,
        td: TensorDict,
        env: FFSPEnv,
        phase="train",
        num_starts=1,
        return_actions: bool = False,
    ):
        assert not env.flatten_stages, "Multistage model only supports unflattened env"
        device = td.device
        td = self.pre_forward(td, env, num_starts)

        batch_size = td.size(0)
        prob_list = torch.zeros(size=(batch_size, 0), device=device)
        action_list = []

        while not td["done"].all():
            action_stack = torch.empty(
                size=(batch_size, self.stage_cnt), dtype=torch.long, device=device
            )
            prob_stack = torch.empty(size=(batch_size, self.stage_cnt), device=device)

            for stage_idx in range(self.stage_cnt):
                model = self.stage_models[stage_idx]
                action, prob = model(td, phase)

                action_stack[:, stage_idx] = action
                prob_stack[:, stage_idx] = prob

            gathering_index = td["stage_idx"][:, None]
            # shape: (batch, 1)
            action = action_stack.gather(dim=1, index=gathering_index).squeeze(dim=1)
            prob = prob_stack.gather(dim=1, index=gathering_index).squeeze(dim=1)
            # shape: (batch)
            action_list.append(action)
            # transition
            td.set("action", action)
            td = env.step(td)["next"]

            prob_list = torch.cat((prob_list, prob[:, None]), dim=1)

        out = {"reward": td["reward"], "log_likelihood": prob_list.log().sum(1)}
        if return_actions:
            out["actions"] = torch.stack(action_list, 1)
        return out


class OneStageModel(nn.Module):
    def __init__(
        self, embedding_dim=256, num_heads=16, num_encoder_layers=3, eval_type="greedy"
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eval_type = eval_type
        # self.encoder = FFSP_Encoder(
        self.encoder = MatNetMHANetwork(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
        )
        self.decoder = FFSP_Decoder()

        self.encoded_col = None
        # shape: (batch, machine_cnt, embedding)
        self.encoded_row = None
        # shape: (batch, job_cnt, embedding)

    def pre_forward(self, cost_mat: torch.Tensor, num_starts=1):
        # problems.shape: (batch, job_cnt, machine_cnt)
        device = cost_mat.device
        batch_size = cost_mat.size(0)
        job_cnt = cost_mat.size(1)
        machine_cnt = cost_mat.size(2)
        embedding_dim = self.embedding_dim

        row_emb = torch.zeros(size=(batch_size, job_cnt, embedding_dim), device=device)
        # shape: (batch, job_cnt, embedding)
        col_emb = torch.zeros(
            size=(batch_size, machine_cnt, embedding_dim), device=device
        )
        # shape: (batch, machine_cnt, embedding)

        rand = torch.rand(batch_size, machine_cnt, device=device)
        batch_rand_perm = rand.argsort(dim=1)
        rand_idx = batch_rand_perm[:, :machine_cnt]

        b_idx = torch.arange(batch_size, device=device)[:, None].expand(
            batch_size, machine_cnt
        )
        m_idx = torch.arange(machine_cnt, device=device)[None, :].expand(
            batch_size, machine_cnt
        )
        col_emb[b_idx, m_idx, rand_idx] = 1
        # shape: (batch, machine_cnt, embedding)

        self.encoded_row, self.encoded_col = self.encoder(row_emb, col_emb, cost_mat)
        if num_starts > 1:
            self.encoded_row = self.encoded_row.repeat(num_starts, 1, 1)
            self.encoded_col = self.encoded_col.repeat(num_starts, 1, 1)

        self.decoder.set_kv(self.encoded_row)

    def forward(self, td: TensorDict, phase="train"):
        device = td.device
        batch_size = td.size(0)
        encoded_current_machine = self.encoded_col.gather(
            1,
            td["stage_machine_idx"][:, None, None].expand(
                batch_size, 1, self.embedding_dim
            ),
        )
        ninf_mask = torch.zeros_like(
            td["action_mask"], dtype=torch.float32, device=device
        ).masked_fill(~td["action_mask"], -torch.inf)

        # shape: (batch, embedding)
        all_job_probs = self.decoder(encoded_current_machine, ninf_mask=ninf_mask)
        # shape: (batch, job)

        if "train" in phase or self.eval_type == "softmax":
            # to fix pytorch.multinomial bug on selecting 0 probability elements
            while True:
                job_selected = all_job_probs.multinomial(1).squeeze(dim=1)
                # shape: (batch)
                job_prob = all_job_probs.gather(1, job_selected[:, None]).squeeze(dim=1)
                # shape: (batch)
                assert (job_prob[td["done"].squeeze()] == 1).all()

                if (job_prob != 0).all():
                    break
        else:
            job_selected = all_job_probs.argmax(dim=1)
            # shape: (batch)
            job_prob = torch.zeros(
                size=(batch_size,), device=device
            )  # any number is okay

        return job_selected, job_prob


########################################
# ENCODER
########################################
class FFSP_Encoder(nn.Module):
    def __init__(self, num_layers=3, **encoder_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(**encoder_kwargs) for _ in range(num_layers)]
        )

    def forward(self, row_emb, col_emb, cost_mat):
        # col_emb.shape: (batch, col_cnt, embedding)
        # row_emb.shape: (batch, row_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**encoder_kwargs)
        self.col_encoding_block = EncodingBlock(**encoder_kwargs)

    def forward(self, row_emb, col_emb, cost_mat):
        # row_emb.shape: (batch, row_cnt, embedding)
        # col_emb.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=16, ff_hidden_dim=512):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_num = num_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(embedding_dim, num_heads)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, ff_hidden_dim)
        self.add_n_normalization_2 = AddAndInstanceNormalization(embedding_dim)

    def forward(self, row_emb, col_emb, cost_mat):
        # NOTE: row and col can be exchanged, if cost_mat.transpose(1,2) is used
        # input1.shape: (batch, row_cnt, embedding)
        # input2.shape: (batch, col_cnt, embedding)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        q = reshape_by_heads(self.Wq(row_emb), head_num=self.head_num)
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        k = reshape_by_heads(self.Wk(col_emb), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=self.head_num)
        # kv shape: (batch, head_num, col_cnt, qkv_dim)

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, row_cnt, embedding)

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, row_cnt, embedding)


# ########################################
# # Decoder
# ########################################


class FFSP_Decoder(nn.Module):
    def __init__(self, embedding_dim=256, head_num=16, logit_clipping=10.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.logit_clipping = logit_clipping
        self.qkv_dim = embedding_dim // head_num

        self.encoded_NO_JOB = nn.Parameter(torch.rand((1, 1, embedding_dim)))

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved key, for single-head attention

    def set_kv(self, encoded_jobs):
        # encoded_jobs.shape: (batch, job, embedding)
        batch_size = encoded_jobs.size(0)

        encoded_no_job = self.encoded_NO_JOB.expand((batch_size, 1, self.embedding_dim))
        encoded_jobs_plus_1 = torch.cat((encoded_jobs, encoded_no_job), dim=1)
        # shape: (batch, job_cnt+1, embedding)

        self.k = reshape_by_heads(self.Wk(encoded_jobs_plus_1), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_jobs_plus_1), head_num=self.head_num)
        # shape: (batch, head_num, job+1, qkv_dim)
        self.single_head_key = encoded_jobs_plus_1.transpose(1, 2)
        # shape: (batch, embedding, job+1)

    def forward(self, encoded_machine, ninf_mask):
        # encoded_machine.shape: (batch, 1, embedding)
        # ninf_mask.shape: (batch, job_cnt+1)

        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq(encoded_machine), head_num=self.head_num)
        # shape: (batch, head_num, qkv_dim)

        out_concat = self._multi_head_attention_for_decoder(
            q, self.k, self.v, ninf_mask=ninf_mask
        )
        # shape: (batch, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key).squeeze(1)
        # shape: (batch, job_cnt+1)

        sqrt_embedding_dim = self.embedding_dim**0.5

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, job_cnt+1)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=1)
        # shape: (batch, job_cnt+1)

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, ninf_mask=None):
        # q shape: (batch, head_num, n, qkv_dim)   : n can be either 1 or PROBLEM_SIZE
        # k,v shape: (batch, head_num, job_cnt+1, qkv_dim)
        # rank2_ninf_mask.shape: (batch, job_cnt+1)
        # rank3_ninf_mask.shape: (batch, n, job_cnt+1)

        batch_size = q.size(0)
        n = q.size(2)
        job_cnt_plus_1 = k.size(2)

        sqrt_qkv_dim = self.qkv_dim**0.5

        score = torch.matmul(q, k.transpose(-2, -1))
        # shape: (batch, head_num, n, job_cnt+1)

        score_scaled = score / sqrt_qkv_dim

        score_scaled = score_scaled + ninf_mask[:, None, None, :].expand(
            batch_size, self.head_num, n, job_cnt_plus_1
        )

        weights = nn.Softmax(dim=3)(score_scaled)
        # shape: (batch, head_num, n, job_cnt+1)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, n, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, n, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, n, self.embedding_dim)
        # shape: (batch, n, head_num*qkv_dim)

        return out_concat


########################################
# NN SUB FUNCTIONS
########################################


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(
            embedding_dim, affine=True, track_running_stats=False
        )

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, embedding_dim=256, ff_hidden_dim=512):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        head_num: int = 16,
        ms_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
    ):
        super().__init__()

        self.head_num = head_num
        self.ms_hidden_dim = ms_hidden_dim
        self.qkv_dim = embedding_dim // head_num

        mix1_weight = torch.torch.distributions.Uniform(
            low=-mix1_init, high=mix1_init
        ).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(
            low=-mix1_init, high=mix1_init
        ).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head, 2, ms_hidden)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head, ms_hidden)

        mix2_weight = torch.torch.distributions.Uniform(
            low=-mix2_init, high=mix2_init
        ).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(
            low=-mix2_init, high=mix2_init
        ).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head, ms_hidden, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head, 1)

    def forward(self, q, k, v, cost_mat):
        # q shape: (batch, head_num, row_cnt, qkv_dim)
        # k,v shape: (batch, head_num, col_cnt, qkv_dim)
        # cost_mat.shape: (batch, row_cnt, col_cnt)

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.head_num
        qkv_dim = self.qkv_dim
        sqrt_qkv_dim = qkv_dim**0.5

        dot_product = torch.matmul(q, k.transpose(2, 3))
        # shape: (batch, head_num, row_cnt, col_cnt)

        dot_product_score = dot_product / sqrt_qkv_dim
        # shape: (batch, head_num, row_cnt, col_cnt)

        cost_mat_score = cost_mat[:, None, :, :].expand(
            batch_size, head_num, row_cnt, col_cnt
        )
        # shape: (batch, head_num, row_cnt, col_cnt)

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_num, row_cnt, col_cnt, 2)

        two_scores_transposed = two_scores.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, col_cnt, 2)

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, ms_hidden_dim)

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, row_cnt, head_num, col_cnt, 1)

        mixed_scores = ms2.transpose(1, 2)
        # shape: (batch, head_num, row_cnt, col_cnt, 1)

        mixed_scores = mixed_scores.squeeze(4)
        # shape: (batch, head_num, row_cnt, col_cnt)

        weights = nn.Softmax(dim=3)(mixed_scores)
        # shape: (batch, head_num, row_cnt, col_cnt)

        out = torch.matmul(weights, v)
        # shape: (batch, head_num, row_cnt, qkv_dim)

        out_transposed = out.transpose(1, 2)
        # shape: (batch, row_cnt, head_num, qkv_dim)

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
        # shape: (batch, row_cnt, head_num*qkv_dim)

        return out_concat
