import torch 
import torch.nn as nn
from rl4co.models.nn.utils import decode_probs


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
        self.logit_attention = LogitAttention()
        self.n_EG = 2

    def forward(self, td, embeddings, decode_type="sampling", softmax_temp=None):
        # SECTION: Decoder first step: calculate for the decoder divergence loss
        # Cost list and log likelihood list along with path
        output_list = []
        td_list = [self.env.reset(td) for i in range(self.num_paths)]
        for i in range(self.num_paths):  
            # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
            fixed = self._precompute(embeddings, path_index=i)
            log_p, _ = self._get_log_p(fixed, td_list[i], i)

            # Collect output of step
            output_list.append(log_p[:, 0, :]) # TODO: for vrp, ignore the first one (depot)
            output_list[-1] = torch.max(output_list[-1], torch.ones(output_list[-1].shape, dtype=output_list[-1].dtype, device=outputs[-1].device) * (-1e9)) # for the kl loss

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
            outputs, actions = [], []
            embeddings, _, attn, V, h_old = self.embedder(self._init_embed(td))
            fixed = self._precompute(embeddings, path_index=i)
            j = 0
            while not (self.shrink_size is None and td_list[i].all_finished()):
                if j > 1 and j % self.eg_step_gap == 0:
                    if not self.is_vrp:
                        mask_attn = mask ^ mask_first
                    else:
                        mask_attn = mask
                    embeddings, _ = self.embedder.change(attn, V, h_old, mask_attn, self.is_tsp)
                    fixed = self._precompute(embeddings, path_index=i)
                log_p, mask = self._get_log_p(fixed, td_list[i], i)
                if j == 0:
                    mask_first = mask

                # Select the indices of the next nodes in the sequences, result (batch_size) long
                # action = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
                action = decode_probs(log_p.exp()[:, 0, :], mask[:, 0, :], decode_type=decode_type)
                td_list[i] = td_list[i].update(action)

                # Collect output of step
                j += 1
                outputs.append(log_p[:, 0, :])
                actions.append(action)

            outputs, actions = torch.stack(outputs, 1), torch.stack(actions, 1)
            reward = self.env.get_reward(td, actions)
            ll = self._calc_log_likelihood(outputs, actions, mask)

            reward_list.append(reward)
            output_list.append(outputs)
            action_list.append(actions)
            ll_list.append(ll)
