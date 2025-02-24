from functools import partial
from typing import Optional, Type, Union

from tensordict import TensorDict
import torch

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.deepaco.antsystem import AntSystem
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.decoding import modify_logits_for_top_k_filtering, modify_logits_for_top_p_filtering
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.utils import merge_with_defaults


class DeepACOPolicy(NonAutoregressivePolicy):
    """Implememts DeepACO policy based on :class:`NonAutoregressivePolicy`. Introduced by Ye et al. (2023): https://arxiv.org/abs/2309.14032.
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
        encoder: Optional[NonAutoregressiveEncoder] = None,
        env_name: str = "tsp",
        temperature: float = 1.0,
        top_p: float = 0.0,
        top_k: int = 0,
        aco_class: Optional[Type[AntSystem]] = None,
        aco_kwargs: dict = {},
        train_with_local_search: bool = False,
        n_ants: Optional[Union[int, dict]] = None,
        n_iterations: Optional[Union[int, dict]] = None,
        start_node: Optional[int] = None,
        multistart: bool = False,
        k_sparse: Optional[int] = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder_kwargs["k_sparse"] = k_sparse
            encoder = NARGNNEncoder(env_name=env_name, **encoder_kwargs)

        self.decode_type = "multistart_sampling" if env_name == "tsp" or multistart else "sampling"

        super().__init__(
            encoder=encoder,
            env_name=env_name,
            temperature=temperature,
            train_decode_type=self.decode_type,
            val_decode_type=self.decode_type,
            test_decode_type=self.decode_type,
        )

        self.default_decoding_kwargs = {}
        self.default_decoding_kwargs["select_best"] = False
        if k_sparse is not None:
            self.default_decoding_kwargs["top_k"] = k_sparse + (0 if env_name == "tsp" else 1)  # 1 for depot
        if "multistart" in self.decode_type:
            select_start_nodes_fn = partial(self.select_start_node_fn, start_node=start_node)
            self.default_decoding_kwargs.update(
                {"multistart": True, "select_start_nodes_fn": select_start_nodes_fn}
            )
        else:
            self.default_decoding_kwargs.update(
                {"multisample": True}
            )

        # For now, top_p and top_k are only used to filter logits (not passed to decoder)
        self.top_p = top_p
        self.top_k = top_k

        self.aco_class = AntSystem if aco_class is None else aco_class
        self.aco_kwargs = aco_kwargs
        self.train_with_local_search = train_with_local_search
        if train_with_local_search:
            assert self.aco_kwargs.get("use_local_search", False)
        self.n_ants = merge_with_defaults(n_ants, train=30, val=48, test=48)
        self.n_iterations = merge_with_defaults(n_iterations, train=1, val=5, test=10)

    @staticmethod
    def select_start_node_fn(
        td: TensorDict, env: RL4COEnvBase, num_starts: int, start_node: Optional[int] = None
    ):
        if env.name == "tsp" and start_node is not None:
            # For now, only TSP supports explicitly setting the start node
            return start_node * torch.ones(
                td.shape[0] * num_starts, dtype=torch.long, device=td.device
            )
        return torch.multinomial(td["action_mask"].float(), num_starts, replacement=True).view(-1)

    def forward(
        self,
        td_initial: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        return_actions: bool = True,
        return_hidden: bool = True,
        actions=None,
        **decoding_kwargs,
    ):
        """
        Forward method. During validation and testing, the policy runs the ACO algorithm to construct solutions.
        See :class:`NonAutoregressivePolicy` for more details during the training phase.
        """
        n_ants = self.n_ants[phase]

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
            #  we just use the constructive policy
            outdict = super().forward(
                td_initial,
                env,
                phase=phase,
                calc_reward=True,
                return_actions=return_actions,
                return_hidden=return_hidden,
                actions=actions,
                **decoding_kwargs,
            )

            outdict["reward"] = unbatchify(outdict["reward"], n_ants)

            if self.train_with_local_search:
                heatmap = outdict["hidden"]
                # TODO: Refactor this so that we don't need to use the aco object
                aco = self.aco_class(heatmap, n_ants=n_ants, **self.aco_kwargs)
                _, ls_reward = aco.local_search(
                    batchify(td_initial, n_ants), env, outdict["actions"], decoding_kwargs
                )
                outdict["ls_reward"] = unbatchify(ls_reward, n_ants)

            outdict["log_likelihood"] = unbatchify(outdict["log_likelihood"], n_ants)
            return outdict

        heatmap, _ = self.encoder(td_initial)
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
