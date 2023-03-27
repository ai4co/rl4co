
import math
from einops import rearrange

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchrl.envs import EnvBase
from tensordict import TensorDict

from ncobench.nn.graph import GraphAttentionEncoder
from ncobench.nn.attention import CrossAttention


class AttentionModelBase(nn.Module):

    def __init__(self,
                 env: EnvBase,
                 embedding_dim: int,
                 hidden_dim: int,
                 *,
                 n_encode_layers: int = 2,
                 tanh_clipping: float = 10.,
                 mask_inner: bool = True,
                 mask_logits: bool = True,
                 normalization: str = 'batch',
                 n_heads: int = 8,
                 checkpoint_encoder: bool = False,
                 use_flash_attn: bool = False,
                 **kwargs
                 ):
        """
        Attention Model base class for neural combinatorial optimization
        Based on Wouter Kool et al. (2018) https://arxiv.org/abs/1803.08475

        Args:
            env: TorchRL Environment
            embedding_dim: Dimension of embedding
            hidden_dim: Dimension of hidden state
            n_encode_layers: Number of encoding layers
            tanh_clipping: Clipping value for tanh
            mask_inner: Whether to mask inner attention
            mask_logits: Whether to mask logits
            normalization: Normalization type
            n_heads: Number of attention heads
            checkpoint_encoder: Whether to use checkpointing for encoder
            use_flash_attn: Whether to use Flash Attention (https://arxiv.org/abs/2205.14135)
        """
        super(AttentionModelBase, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.env = env

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder

        # TODO: add extra except TSP
        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        node_dim = 2  # x, y
        
        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
            use_flash_attn=use_flash_attn,
        )
        
        self.cross_attention = CrossAttention() # NOTE: FlashCrossAttention does not support inner masking

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, td: TensorDict, phase: str = "train", decode_type: str = "sampling") -> TensorDict:
        """Given observation, precompute embeddings and rollout"""

        # Set decoding type for policy, can be also greedy
        self.decode_type = decode_type

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(td['observation']))
        else:
            embeddings, _ = self.embedder(self._init_embed(td['observation']))

        # Main rollout
        _log_p, actions, td = self._rollout(td, embeddings)
        reward = self.env.get_rewards(td['observation'], actions)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        ll = self._calc_log_likelihood(_log_p, actions, td.get('mask', None))
        out = {"reward": reward, "log_likelihood": ll, "actions": actions, "cost": -reward}
        return out
    
    def _rollout(self, td, embeddings):

        outputs = []
        actions = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Here we could use the env.unroll() method as well but we would need to collec
        while not td["done"].any():
            
            log_p, mask = self._get_log_p(fixed, td)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            action = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
           
           # Set action and step environment
            td.set("action", action[:, None])
            td = self.env.step(td)['next']

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            actions.append(action)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(actions, 1), td

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, x):
        # TODO: others except TSP
        return self.init_embed(x)
    
    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected
    
    def _precompute(self, embeddings, num_steps=1):
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
        
        # Organize in a TensorDict for easy access
        fixed = TensorDict({
                "node_embeddings": embeddings,
                "context_node_projected": self.project_fixed_context(graph_embed)[:, None, :],
                "glimpse_key": self._make_heads(glimpse_key_fixed, num_steps),
                "glimpse_val": self._make_heads(glimpse_val_fixed, num_steps),
                "logit_key": logit_key_fixed.contiguous()
            },
            batch_size=[], # no batch dimension since we are only storing the fixed data
        )
        return fixed
        
    def _get_log_p(self, fixed, td, normalize=True):
        
        # Compute query = context node embedding
        query = fixed["context_node_projected"] + \
                self.project_step_context(self._get_parallel_step_context(fixed["node_embeddings"], td))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, td['observation'])

        # Compute the mask
        mask = self.env.get_mask(td)

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, td):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        """
        current_node = self.env.get_current_node(td)
        batch_size, num_steps = current_node.size()

        # TODO: add others except TSP
        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if td['i'][0].item() == 0: 
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((td['first_a'], current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
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

    def _one_to_many_logits(self, query, key, value, logit_K, mask):
        # Rearranging
        kv = torch.stack([key, value])
        q = rearrange(query, 'b 1 (h s) -> b 1 h s', h=self.n_heads)
        kv = rearrange(kv, 'two h b 1 g s -> b g two h s', two=2, h=self.n_heads)     

        # 1 means to keep, so we invert the mask
        key_padding_mask = ~mask.squeeze()

        # Cross attention and projection to get glimpse
        heads = self.cross_attention(q, kv, key_padding_mask=key_padding_mask)
        heads = rearrange(heads, 'b 1 h g -> b 1 1 (h g)', h=self.n_heads)
        glimpse = self.project_out(heads)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits

    def _get_attention_node_data(self, fixed: TensorDict, td: TensorDict) -> dict:
        # TODO: add others except TSP
        return fixed["glimpse_key"], fixed["glimpse_val"], fixed["logit_key"]

    def _make_heads(self, v, num_steps=None):
        # TODO: refactor so no need for rearrange
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )