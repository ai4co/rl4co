from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.nn.attention import LogitAttention
from rl4co.models.nn.dec_strategies import DecodingStrategy, get_decoding_strategy
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.utils import get_log_likelihood
from rl4co.utils.ops import batchify, select_start_nodes, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@dataclass
class PrecomputedCache:
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor


class AutoregressiveDecoder(nn.Module):
    """Auto-regressive decoder for constructing solutions for combinatorial optimization problems.
    Given the environment state and the embeddings, compute the logits and sample actions autoregressively until
    all the environments in the batch have reached a terminal state.
    We additionally include support for multi-starts as it is more efficient to do so in the decoder as we can
    natively perform the attention computation.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward is not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        env_name: environment name to solve
        embedding_dim: Dimension of the embeddings
        num_heads: Number of heads for the attention
        use_graph_context: Whether to use the initial graph context to modify the query
        select_start_nodes_fn: Function to select the start nodes for multi-start decoding
        linear_bias: Whether to use a bias in the linear projection of the embeddings
        context_embedding: Module to compute the context embedding. If None, the default is used
        dynamic_embedding: Module to compute the dynamic embedding. If None, the default is used
    """

    def __init__(
        self,
        env_name: Union[str, RL4COEnvBase],
        embedding_dim: int,
        num_heads: int,
        use_graph_context: bool = True,
        select_start_nodes_fn: callable = select_start_nodes,
        linear_bias: bool = False,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        **logit_attn_kwargs,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embedding_dim": embedding_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        self.use_graph_context = use_graph_context

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=linear_bias
        )
        self.project_fixed_context = nn.Linear(
            embedding_dim, embedding_dim, bias=linear_bias
        )

        # MHA
        self.logit_attention = LogitAttention(
            embedding_dim, num_heads, **logit_attn_kwargs
        )

        self.select_start_nodes_fn = select_start_nodes_fn

    def forward(
        self,
        td: TensorDict,
        embeddings: Tensor,
        env: Union[str, RL4COEnvBase] = None,
        decode_type: str = "sampling",
        softmax_temp: float = None,
        calc_reward: bool = True,
        **strategy_kwargs,
    ) -> Tuple[Tensor, Tensor, TensorDict]:
        """Forward pass of the decoder
        Given the environment state and the pre-computed embeddings, compute the logits and sample actions

        Args:
            td: Input TensorDict containing the environment state
            embeddings: Precomputed embeddings for the nodes
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            decode_type: Type of decoding to use. Can be one of:
                - "sampling": sample from the logits
                - "greedy": take the argmax of the logits
                - "multistart_sampling": sample as sampling, but with multi-start decoding
                - "multistart_greedy": sample as greedy, but with multi-start decoding
                - "beam_search": perform beam search
            softmax_temp: Temperature for the softmax. If None, default softmax is used from the `LogitAttention` module
            calc_reward: Whether to calculate the reward for the decoded sequence
            strategy_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.models.nn.dec_strategies.DecodingStrategy`

        Returns:
            outputs: Tensor of shape (batch_size, seq_len, num_nodes) containing the logits
            actions: Tensor of shape (batch_size, seq_len) containing the sampled actions
            td: TensorDict containing the environment state after decoding
        """
        # Instantiate environment if needed
        if isinstance(env, str):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute_cache(embeddings, td=td)

        # If `select_start_nodes_fn` is not being passed, we use the class attribute
        if "select_start_nodes_fn" not in strategy_kwargs:
            strategy_kwargs["select_start_nodes_fn"] = self.select_start_nodes_fn

        # Setup decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type, **strategy_kwargs
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Main decoding: loop until all sequences are done
        while not td["done"].all():
            log_p, mask = self._get_log_p(cached_embeds, td, softmax_temp, num_starts)
            td = decode_strategy.step(log_p, mask, td)
            td = env.step(td)["next"]

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        outputs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        return outputs, actions, td

    def _precompute_cache(
        self,
        embeddings: Tensor,
        td: TensorDict = None,
    ):
        """Compute the cached embeddings for the attention

        Args:
            embeddings: Precomputed embeddings for the nodes
            td: TensorDict containing the environment state.
            This one is not used in this class. However, passing Tensordict can be useful in child classes.
        """

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        cached_embeds = PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

        return cached_embeds

    def _get_log_p(
        self,
        cached: PrecomputedCache,
        td: TensorDict,
        softmax_temp: float = None,
        num_starts: int = 0,
    ):
        """Compute the log probabilities of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            softmax_temp: Temperature for the softmax
            num_starts: Number of starts for the multi-start decoding
        """

        # Get precomputed (cached) embeddings
        node_embeds_cache, graph_context_cache = (
            cached.node_embeddings,
            cached.graph_context,
        )
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key,
            cached.glimpse_val,
            cached.logit_key,
        )  # [B, N, H]
        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            glimpse_k_stat = batchify(glimpse_k_stat, num_starts)
            glimpse_v_stat = batchify(glimpse_v_stat, num_starts)
            logit_k_stat = batchify(logit_k_stat, num_starts)
            node_embeds_cache = batchify(node_embeds_cache, num_starts)
            graph_context_cache = (
                batchify(graph_context_cache, num_starts)
                if isinstance(graph_context_cache, Tensor)
                else graph_context_cache
            )
        elif num_starts > 1:
            td = unbatchify(td, num_starts)
            if isinstance(graph_context_cache, Tensor):
                # add a dimension for num_starts (will automatically be broadcasted during addition)
                graph_context_cache = graph_context_cache.unsqueeze(1)

        step_context = self.context_embedding(node_embeds_cache, td)
        glimpse_q = step_context + graph_context_cache
        glimpse_q = (
            glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        )  # add seq_len dim if not present

        # Compute dynamic embeddings and add to static embeddings
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn

        # Get the mask
        mask = ~td["action_mask"]

        # Compute logits
        log_p = self.logit_attention(
            glimpse_q, glimpse_k, glimpse_v, logit_k, mask, softmax_temp
        )

        # Now we need to reshape the logits and log_p to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            log_p = rearrange(log_p, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return log_p, mask

    def evaluate_action(
        self,
        td: TensorDict,
        embeddings: Tensor,
        action: Tensor,
        env: Union[str, RL4COEnvBase] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Evaluate the (old) action to compute
        log likelihood of the actions and corresponding entropy
        Args:
            td: Input TensorDict containing the environment state
            embeddings: Precomputed embeddings for the nodes
            action: Action to evaluate (batch_size, seq_len)
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
        Returns:
            log_p: Tensor of shape (batch_size, seq_len, num_nodes) containing the log-likehood of the actions
            entropy: Tensor of shape (batch_size, seq_len) containing the sampled actions
        """

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            env = get_env(env_name)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        cached_embeds = self._precompute_cache(embeddings)

        log_p = []
        decode_step = 0
        while not td["done"].all():
            log_p_, _ = self._get_log_p(cached_embeds, td)
            action_ = action[..., decode_step]

            td.set("action", action_)
            td = env.step(td)["next"]
            log_p.append(log_p_)

            decode_step += 1

        # Note that the decoding steps may not be equal to the decoding steps of actions
        # due to the padded zeros in the actions

        # Compute log likelihood of the actions
        log_p = torch.stack(log_p, 1)  # [batch_size, decoding steps, num_nodes]
        ll = get_log_likelihood(
            log_p, action[..., :decode_step], mask=None, return_sum=False
        )  # [batch_size, decoding steps]
        assert ll.isfinite().all(), "Log p is not finite"

        # compute entropy
        log_p = torch.nan_to_num(log_p, nan=0.0)
        entropy = -(log_p.exp() * log_p).sum(dim=-1)  # [batch, decoder steps]
        entropy = entropy.sum(dim=1)  # [batch] -- sum over decoding steps
        assert entropy.isfinite().all(), "Entropy is not finite"

        return ll, entropy
