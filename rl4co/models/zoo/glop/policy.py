from typing import Literal, Optional, Union

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
from rl4co.models.zoo.glop.utils import eval_insertion
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.ops import batchify, select_start_nodes_by_distance
from rl4co.utils.pylogger import get_pylogger

try:
    from rl4co.models.zoo.glop.adapter import SubTSPAdapter
except ImportError:
    # In case some dependencies are not installed (e.g., numba)
    SubTSPAdapter = None

log = get_pylogger(__name__)


class GLOPPolicy(NonAutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        env_name: str = "cvrp",
        n_samples: int = 10,
        temperature: float = 0.1,
        **encoder_kwargs,
    ):
        assert (
            SubTSPAdapter is not None
        ), "Cannot import adapter module. Please check if `numba` is installed."

        if encoder is None:
            embed_dim = encoder_kwargs.get("embed_dim", 64)
            if "init_embedding" not in encoder_kwargs:
                encoder_kwargs["init_embedding"] = VRPPolarInitEmbedding(
                    embed_dim, attach_cartesian_coords=True
                )
            if "edge_embedding" not in encoder_kwargs:
                encoder_kwargs["edge_embedding"] = VRPPolarEdgeEmbedding(embed_dim)
            encoder = NARGNNEncoder(**encoder_kwargs)
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

        SubTSPAdapter.pre_compile_numba()

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
        subtsp_solver: Literal["am", "insertion", "lkh"] = "am",
        actions=None,
        **decoding_kwargs,
    ) -> dict:
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
            actions=actions,
            **decoding_kwargs,
        )

        # local policy
        par_actions = par_out["actions"]
        local_policy_out = self.local_policy(
            td,
            par_actions,
            phase=phase,
            subtsp_solver=subtsp_solver,
        )

        # Construct final output
        out = par_out

        if calc_reward:
            with torch.no_grad():
                if isinstance(env, str) or env is None:
                    env_name = self.env_name if env is None else env
                    log.info(
                        f"Instantiated environment not provided; instantiating {env_name}"
                    )
                    env = get_env(env_name)
                td_repeated = batchify(td, self.n_samples)
                reward = env.get_reward(
                    td_repeated, local_policy_out["actions"], check_solution=False
                )
                out["reward"] = reward.detach()

        out["actions"] = local_policy_out["actions"]

        return out

    @torch.no_grad()
    def local_policy(
        self,
        td: TensorDict,
        actions: torch.Tensor,
        /,
        phase: Literal["train", "val", "test"] = "train",
        subtsp_solver: Literal["am", "insertion", "lkh"] = "am",
    ):
        assert SubTSPAdapter is not None
        actions = rearrange(actions, "(n b) ... -> (b n) ...", n=self.n_samples)

        adapter = SubTSPAdapter(td, actions)
        for mapping in adapter.get_batched_subtsps(batch_size=None):
            subtsp_actions, _ = eval_insertion(mapping.subtsp_coordinates)
            adapter.update_actions(mapping, subtsp_actions)

        actions_revised = adapter.get_actions().to(td.device)
        actions_revised = rearrange(actions_revised, "(b n) ... -> (n b) ...", n=self.n_samples)
        return dict(actions=actions_revised)
