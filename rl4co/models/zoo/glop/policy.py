from typing import Callable, Literal, Optional, Union

import torch

from einops import rearrange
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveDecoder,
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.models.nn.env_embeddings.edge import VRPPolarEdgeEmbedding
from rl4co.models.nn.env_embeddings.init import VRPPolarInitEmbedding
from rl4co.models.zoo.glop.adapter import adapter_map
from rl4co.models.zoo.glop.utils import eval_insertion, eval_lkh
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.ops import batchify, select_start_nodes_by_distance
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

SubTSPSolverType = Union[
    Literal["insertion", "lkh"],
    Callable[[torch.Tensor], torch.Tensor],
]


class GLOPPolicy(NonAutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        env_name: str = "cvrp",
        n_samples: int = 10,
        temperature: float = 0.1,
        subtsp_adapter_class=None,
        subtsp_solver: SubTSPSolverType = "insertion",
        subtsp_batchsize: int = 1000,
        **encoder_kwargs,
    ):

        if subtsp_adapter_class is None:
            # TODO: test more VRPs
            assert (
                env_name in adapter_map
            ), f"{env_name} is not supported by {self.__class__.__name__} yet"
            subtsp_adapter_class = adapter_map.get(env_name)
            assert (
                subtsp_adapter_class is not None
            ), "Can not import adapter module. Please check if `numba` is installed."

        if encoder is None:
            encoder_kwargs.setdefault("embed_dim", 64)
            if env_name.startswith("cvrp"):
                embed_dim = encoder_kwargs.get("embed_dim", 64)
                if "init_embedding" not in encoder_kwargs:
                    encoder_kwargs["init_embedding"] = VRPPolarInitEmbedding(
                        embed_dim, attach_cartesian_coords=True
                    )
                if "edge_embedding" not in encoder_kwargs:
                    encoder_kwargs["edge_embedding"] = VRPPolarEdgeEmbedding(embed_dim)
            encoder = NARGNNEncoder(env_name=env_name, **encoder_kwargs)
        if decoder is None:
            decoder = NonAutoregressiveDecoder()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_greedy",
            test_decode_type="multistart_greedy",
        )

        self.n_samples = n_samples
        self.subtsp_solver: SubTSPSolverType = subtsp_solver
        self.subtsp_adapter_class = subtsp_adapter_class
        self.subtsp_batchsize = subtsp_batchsize

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[RL4COEnvBase, str]] = None,
        phase: Literal["train", "val", "test"] = "test",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        subtsp_solver: Optional[SubTSPSolverType] = None,
        **decoding_kwargs,
    ) -> dict:
        if (
            env is None
            and self.env_name.startswith("cvrp")
            or isinstance(env, str)
            and env.startswith("cvrp")
            or isinstance(env, RL4COEnvBase)
            and env.name.startswith("cvrp")
        ):
            decoding_kwargs.setdefault(
                "select_start_nodes_fn", select_start_nodes_by_distance
            )

        par_out = super().forward(
            td=td,
            env=env,
            phase=phase,
            calc_reward=False,  # We don't need the partition reward
            return_actions=True,  # Used for partition
            return_entropy=return_entropy,
            return_init_embeds=return_init_embeds,
            return_sum_log_likelihood=return_sum_log_likelihood,
            num_starts=self.n_samples,
            **decoding_kwargs,
        )

        subtsp_solver = self.subtsp_solver if subtsp_solver is None else subtsp_solver
        if isinstance(subtsp_solver, str):
            if subtsp_solver == "insertion":
                subtsp_solver = eval_insertion
            elif subtsp_solver == "lkh":
                subtsp_solver = eval_lkh
            else:
                raise ValueError(f"Unexpected sub-TSP solver value '{subtsp_solver}'")

        # local policy
        par_actions = par_out["actions"]
        local_policy_out = self.local_policy(
            td,
            par_actions,
            subtsp_solver=subtsp_solver,
        )
        actions = local_policy_out["actions"]

        # Construct final output
        out = par_out

        if return_actions:
            out["actions"] = actions
        else:
            del out["actions"]

        if calc_reward:
            if isinstance(env, str) or env is None:
                env_name = self.env_name if env is None else env
                log.info(
                    f"Instantiated environment not provided; instantiating {env_name}"
                )
                env = get_env(env_name)
            td_repeated = batchify(td, self.n_samples)
            reward = env.get_reward(td_repeated, actions)
            out["reward"] = reward.detach()
        return out

    @torch.no_grad()
    def local_policy(
        self,
        td: TensorDict,
        actions: torch.Tensor,
        subtsp_solver,
    ):
        assert self.subtsp_adapter_class is not None
        actions = rearrange(actions, "(n b) ... -> (b n) ...", n=self.n_samples)

        adapter = self.subtsp_adapter_class(td, actions)
        for mapping in adapter.get_batched_subtsps(batch_size=self.subtsp_batchsize):
            subtsp_actions = subtsp_solver(mapping.subtsp_coordinates)
            adapter.update_actions(mapping, subtsp_actions)

        actions_revised = adapter.get_actions().to(td.device)
        actions_revised = rearrange(
            actions_revised, "(b n) ... -> (n b) ...", n=self.n_samples
        )
        return dict(actions=actions_revised)
