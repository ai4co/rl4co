from typing import Optional, Type, Union

from tensordict import TensorDict
import torch

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.zoo.deepaco import DeepACOPolicy
from rl4co.models.zoo.deepaco.antsystem import AntSystem
from rl4co.models.zoo.gfacs.encoder import GFACSEncoder
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
    modify_logits_for_top_k_filtering,
    modify_logits_for_top_p_filtering
)
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class GFACSPolicy(DeepACOPolicy):
    """Implememts GFACS policy based on :class:`NonAutoregressivePolicy`. Introduced by Kim et al. (2024): https://arxiv.org/abs/2403.07041.
    This policy uses a Non-Autoregressive Graph Neural Network to generate heatmaps,
    which are then used to run Ant Colony Optimization (ACO) to construct solutions.

    Args:
        encoder: Encoder module. Can be passed by sub-classes
        env_name: Name of the environment used to initialize embeddings
        temperature: Temperature for the softmax during decoding. Defaults to 1.0.
        aco_class: Class representing the ACO algorithm to be used. Defaults to :class:`AntSystem`.
        aco_kwargs: Additional arguments to be passed to the ACO algorithm.
        train_with_local_search: Whether to train with local search. Defaults to False.
        n_ants: Number of ants to be used in the ACO algorithm. Can be an integer or dictionary. Defaults to 20.
        n_iterations: Number of iterations to run the ACO algorithm. Can be an integer or dictionary. Defaults to `dict(train=1, val=20, test=100)`.
        encoder_kwargs: Additional arguments to be passed to the encoder.
    """

    def __init__(
        self,
        encoder: Optional[GFACSEncoder] = None,
        env_name: str = "tsp",
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        aco_class: Optional[Type[AntSystem]] = None,
        aco_kwargs: dict = {},
        train_with_local_search: bool = True,
        n_ants: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        multistart: bool = False,
        k_sparse: Optional[int] = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder_kwargs["z_out_dim"] = 2 if train_with_local_search else 1
            encoder_kwargs["k_sparse"] = k_sparse
            encoder = GFACSEncoder(env_name=env_name, **encoder_kwargs)

        super().__init__(
            encoder=encoder,
            env_name=env_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            aco_class=aco_class,
            aco_kwargs=aco_kwargs,
            train_with_local_search=train_with_local_search,
            n_ants=n_ants,
            n_iterations=n_iterations,
            multistart=multistart,
            k_sparse=k_sparse,
        )

    def forward(
        self,
        td_initial: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        return_actions: bool = True,
        return_hidden: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        """
        Forward method. During validation and testing, the policy runs the ACO algorithm to construct solutions.
        See :class:`NonAutoregressivePolicy` for more details during the training phase.
        """
        n_ants = self.n_ants[phase]

        heatmap, _, logZ = self.encoder(td_initial)

        decoding_kwargs.update(self.default_decoding_kwargs)
        decoding_kwargs.update(
            {"num_starts": n_ants} if "multistart" in self.decode_type else {"num_samples": n_ants}
        )

        # Instantiate environment if needed
        if (phase != "train" or self.train_with_local_search) and (env is None or isinstance(env, str)):
            env_name = self.env_name if env is None else env
            env = get_env(env_name)
        else:
            assert isinstance(env, RL4COEnvBase), "env must be an instance of RL4COEnvBase"

        if phase == "train":
            # Encoder: get encoder output and initial embeddings from initial state
            if self.train_with_local_search:
                logZ, ls_logZ = logZ[:, [0]], logZ[:, [1]]
            else:
                logZ = logZ[:, [0]]

            logprobs, actions, td, env = self.common_decoding(
                self.decode_type, td_initial, env, heatmap, actions, **decoding_kwargs
            )

            # Output dictionary construction
            outdict = {
                "logZ": logZ,
                "reward": unbatchify(env.get_reward(td, actions), n_ants),
                "log_likelihood": unbatchify(
                    get_log_likelihood(logprobs, actions, td.get("mask", None), True), n_ants
                )
            }

            if return_actions:
                outdict["actions"] = actions

            ########################################################################
            # Local search
            if self.train_with_local_search:
                # TODO: Refactor this so that we don't need to use the aco object
                aco = self.aco_class(heatmap, n_ants=n_ants, **self.aco_kwargs)
                ls_actions, ls_reward = aco.local_search(
                    batchify(td_initial, n_ants), env, actions, decoding_kwargs
                )
                ls_decoding_kwargs = decoding_kwargs.copy()
                ls_decoding_kwargs["top_k"] = 0  # This should be 0, otherwise logprobs can be -inf
                ls_logprobs, ls_actions, td, env = self.common_decoding(
                    "evaluate", td_initial, env, heatmap, ls_actions, **ls_decoding_kwargs
                )
                outdict.update(
                    {
                        "ls_logZ": ls_logZ,
                        "ls_reward": unbatchify(ls_reward, n_ants),
                        "ls_log_likelihood": unbatchify(
                            get_log_likelihood(ls_logprobs, ls_actions, td.get("mask", None), True),
                            n_ants,
                        )
                    }
                )
                if return_actions:
                    outdict["ls_actions"] = ls_actions
            ########################################################################

            if return_hidden:
                outdict["hidden"] = heatmap

            return outdict

        heatmap /= self.temperature

        if self.top_k > 0:
            self.top_k = min(self.top_k, heatmap.size(-1))  # safety check
            heatmap = modify_logits_for_top_k_filtering(heatmap, self.top_k)

        if self.top_p > 0:
            assert self.top_p <= 1.0, "top-p should be in (0, 1]."
            heatmap = modify_logits_for_top_p_filtering(heatmap, self.top_p)

        aco = self.aco_class(heatmap, n_ants=n_ants, **self.aco_kwargs)
        actions, iter_rewards = aco.run(td_initial, env, self.n_iterations[phase], decoding_kwargs)

        out = {"reward": iter_rewards[self.n_iterations[phase] - 1]}
        out.update({f"reward_{i:03d}": iter_rewards[i] for i in range(self.n_iterations[phase])})
        if return_actions:
            out["actions"] = actions

        return out

    def common_decoding(
        self,
        decode_type: str | DecodingStrategy,
        td: TensorDict,
        env: RL4COEnvBase,
        hidden: TensorDict,
        actions: Optional[torch.Tensor] = None,
        max_steps: int = 1_000_000,
        **decoding_kwargs,
    ):
        decoding_strategy: DecodingStrategy = get_decoding_strategy(
            decoding_strategy=decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            select_start_nodes_fn=decoding_kwargs.pop("select_start_nodes_fn", None),
            store_all_logp=decoding_kwargs.pop("store_all_logp", False),
            **decoding_kwargs,
        )
        if actions is not None:
            assert decoding_strategy.name == "evaluate", "decoding strategy must be 'evaluate' when actions are provided"

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decoding_strategy.pre_decoder_hook(
            td, env, actions[:, 0] if actions is not None and "multistart" in self.decode_type else None
        )

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main decoding: loop until all sequences are done
        step = 1 if "multistart" in self.decode_type else 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decoding_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decoding_strategy.post_decoder_hook(td, env)
        return logprobs, actions, td, env