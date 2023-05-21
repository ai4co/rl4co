import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.models.nn.env_context import env_context
from rl4co.models.nn.env_embedding import env_dynamic_embedding
from rl4co.models.nn.attention import LogitAttention


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
            num_paths: int = 5,
            mask_inner: bool = True,
            mask_logits: bool = True,
            eg_step_gap: int = 200,
            tanh_clipping: float = 10.0,
            force_flash_attn: bool = False,
            shrink_size=None,
            train_decode_type: str = "sampling",
            val_decode_type: str = "greedy",
            test_decode_type: str = "greedy",
        ):
        super(Decoder, self).__init__()
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type
        
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.context = [env_context(env.name, {"embedding_dim": embedding_dim}) for _ in range(num_paths)]

        self.project_node_embeddings = [nn.Linear(embedding_dim, 3 * embedding_dim, device=env.device, bias=False) for _ in range(num_paths)]
        self.project_node_embeddings = nn.ModuleList(self.project_node_embeddings)

        self.project_fixed_context = [nn.Linear(embedding_dim, embedding_dim, device=env.device, bias=False) for _ in range(num_paths)]
        self.project_fixed_context = nn.ModuleList(self.project_fixed_context)
        
        self.project_step_context = [nn.Linear(2 * embedding_dim, embedding_dim, device=env.device, bias=False) for _ in range(num_paths)]
        self.project_step_context = nn.ModuleList(self.project_step_context)

        self.project_out = [nn.Linear(embedding_dim, embedding_dim, device=env.device, bias=False) for _ in range(num_paths)]
        self.project_out = nn.ModuleList(self.project_out)

        self.dynamic_embedding = env_dynamic_embedding(
            env.name, {"embedding_dim": embedding_dim}
        )

        self.logit_attention = LogitAttention(
            embedding_dim, 
            num_heads, 
            mask_inner=mask_inner,
            force_flash_attn=force_flash_attn,
        )

        self.env = env
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.num_heads = num_heads
        self.num_paths = num_paths
        self.eg_step_gap = eg_step_gap
        self.tanh_clipping = tanh_clipping
        self.shrink_size = shrink_size

    def forward(
            self, 
            td, 
            encoded_inputs, 
            attn, 
            V, 
            h_old,
            **decoder_kwargs
        ):
        # SECTION: Decoder first step: calculate for the decoder divergence loss
        # Cost list and log likelihood list along with path
        output_list = []
        td_list = [self.env.reset(td) for i in range(self.num_paths)]
        for i in range(self.num_paths):  
            # Clone the encoded features for this path
            _encoded_inputs = encoded_inputs.clone()

            # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
            fixed = self._precompute(_encoded_inputs, path_index=i)
            log_p, _ = self._get_log_p_temp(fixed, td_list[i], i)

            # Collect output of step
            output_list.append(log_p[:, 0, :]) # TODO: for vrp, ignore the first one (depot)
            output_list[-1] = torch.max(output_list[-1], torch.ones(output_list[-1].shape, dtype=output_list[-1].dtype, device=output_list[-1].device) * (-1e9)) # for the kl loss

        if self.num_paths > 1: # TODO: add a check for the baseline
            kl_divergences = []
            for _i in range(self.num_paths):
                for _j in range(self.num_paths):
                    if _i==_j:
                        continue
                    kl_divergence = torch.sum(torch.exp(output_list[_i]) * (output_list[_i] - output_list[_j]), -1)
                    kl_divergences.append(kl_divergence)
            loss_kl_divergence = torch.stack(kl_divergences, 0).mean()

        # SECTION: Decoder rest step: calculate for other decoder divergence loss
        # Cost list and log likelihood list along with path
        reward_list = []; output_list = []; action_list = []; ll_list = []
        td_list = [self.env.reset(td) for _ in range(self.num_paths)]
        for i in range(self.num_paths):
            # Clone the encoded features for this path
            _encoded_inputs = encoded_inputs.clone()
            _attn = attn.clone()
            _V = V.clone()
            _h_old = h_old.clone()

            outputs, actions = [], []
            fixed = self._precompute(_encoded_inputs, path_index=i)

            j = 0
            while not (self.shrink_size is None and td_list[i]["done"].all()):
                if j > 1 and j % self.eg_step_gap == 0:
                    if not self.is_vrp:
                        mask_attn = mask ^ mask_first
                    else:
                        mask_attn = mask
                    _encoded_inputs, _ = self.embedder.change(_attn, _V, _h_old, mask_attn, self.is_tsp)
                    fixed = self._precompute(_encoded_inputs, path_index=i)
                log_p, mask = self._get_log_p_temp(fixed, td_list[i], i)
                if j == 0:
                    mask_first = mask

                # Select the indices of the next nodes in the sequences, result (batch_size) long
                action = self._select_node(log_p.exp()[:, 0, :], mask, decode_type=decoder_kwargs["decode_type"])

                td_list[i].set("action", action)
                td_list[i] = self.env.step(td_list[i])["next"]

                # Collect output of step
                outputs.append(log_p[:, 0, :])
                actions.append(action)
                j += 1

            outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
            reward = self.env.get_reward(td, actions)
            ll = get_log_likelihood(outputs, actions, mask)

            reward_list.append(reward)
            output_list.append(outputs)
            action_list.append(actions)
            ll_list.append(ll)

        reward = torch.stack(reward_list, 0)
        log_likelihood = torch.stack(ll_list, 0)
        return reward, log_likelihood, loss_kl_divergence, actions

    def _select_node(self, probs, mask, decode_type="sampling"):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
        elif decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1, path_index=None):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)

        # Fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context[path_index](graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings[path_index](embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=fixed_context,
            glimpse_key=self._make_heads(glimpse_key_fixed, num_steps),
            glimpse_val=self._make_heads(glimpse_val_fixed, num_steps),
            logit_key=logit_key_fixed.contiguous(),
        )
        return fixed

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.num_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.num_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    def _get_log_p(self, cached, td, path_idx, normalize=True):
        step_context = self.context[path_idx](cached.node_embeddings, td).to(td.device)  # [batch, embed_dim]
        glimpse_q = (cached.graph_context + step_context).unsqueeze(1)  # [batch, 1, embed_dim]

        # Compute keys and values for the nodes
        (
            glimpse_key_dynamic,
            glimpse_val_dynamic,
            logit_key_dynamic,
        ) = self.dynamic_embedding(td)
        glimpse_k = cached.glimpse_key + glimpse_key_dynamic
        glimpse_v = cached.glimpse_val + glimpse_val_dynamic
        logit_k = cached.logit_key + logit_key_dynamic

        # Get the mask
        mask = ~td["action_mask"]
        
        # Compute log prob: MHA + single-head attention
        log_p, _ = self._one_to_many_logits(
            glimpse_q,
            glimpse_k,
            glimpse_v,
            logit_k,
            mask,
            path_idx
        )

        if normalize:
            log_p = F.log_softmax(log_p / 1., dim=-1) # TODO

        return log_p, mask

    def _get_log_p_temp(self, fixed, td, path_index, normalize=True):
        # Compute query = context node embedding
        query = fixed.graph_context + \
                self.project_step_context[path_index](self._get_parallel_step_context(fixed.node_embeddings, td))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, td)

        # Compute the mask
        mask = ~td["action_mask"]

        # Compute logits (unnormalized log_p)
        log_p, _ = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, path_index)

        if normalize:
            log_p = F.log_softmax(log_p / 1., dim=-1) # TODO

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_attention_node_data(self, fixed, td):
        # TODO: for vrp
        # glimpse_key_step, glimpse_val_step, logit_key_step = \
        #     self.project_node_step(td.demands[:, :, :, None].clone()).chunk(3, dim=-1)
        # return (
        #     fixed.glimpse_key + self._make_heads(glimpse_key_step),
        #     fixed.glimpse_val + self._make_heads(glimpse_val_step),
        #     fixed.logit_key + logit_key_step,
        # )
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _get_parallel_step_context(self, embeddings, td, from_depot=False):
        current_node = td["current_node"][:, None]
        batch_size, num_steps = current_node.size()

        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if td["i"][0].item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((td["first_node"][:, None], current_node), -1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, cat in dim 1 (time steps)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, path_index):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.num_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.num_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, None, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out[path_index](
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.num_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask[:, None, :]] = -math.inf

        return logits, glimpse.squeeze(-2)