from typing import Literal, Optional, Union

import numpy as np
import torch

from einops import rearrange
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveDecoder,
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.models.nn.env_embeddings.edge import VRPPolarEdgeEmbedding
from rl4co.models.nn.env_embeddings.init import VRPPolarInitEmbedding
from rl4co.models.zoo.glop.utils import cvrp_to_subtsp, get_total_cost
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder
from rl4co.utils.ops import select_start_nodes_by_distance
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GLOPPolicy(NonAutoregressivePolicy):
    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        env_name: str = "cvrp",
        n_samples: int = 10,
        opts: list[Union[callable]] = None,
        **encoder_kwargs,
    ):
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
            temperature=0.1,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_greedy",
            test_decode_type="multistart_greedy",
        )

        self.n_samples = n_samples
        self.opts = opts

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        phase: Literal["train", "val", "test"] = "train",
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

        par_actions = par_out["actions"]
        par_log_likelihood = par_out["log_likelihood"]

        local_policy_out = self.local_policy(
            td,
            par_actions,
            phase=phase,
            subtsp_solver=subtsp_solver,
        )

        # Construct final output
        out = {
            "log_likelihood": par_log_likelihood,
            "reward": local_policy_out["reward"],
        }

        # if return_additional_info:
        #     out.update({
        #         "par_actions": rearrange(par_actions.cpu(), "(b n) ... -> (n b) ...", n=self.n_samples),
        #         # "tsp_insts": tsp_insts_list,
        #         # "n_tsps_per_route": n_tsps_per_route_list,
        #         # "tours_list": tours_list,
        #     })

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
        device = td.device

        actions = rearrange(actions, "(n b) ... -> (b n) ...", n=self.n_samples)

        # Based on partition actions to get partitions
        tsp_insts_list, n_tsps_per_route_list = self.partition_glop(td, actions, phase)

        reward_list = []
        tours_list = []
        for batch_idx in range(td.batch_size[0]):
            tsp_insts = tsp_insts_list[batch_idx]
            n_tsps_per_route = n_tsps_per_route_list[batch_idx]
            if phase == "train" or subtsp_solver == "insertion":
                objs = here_eval(tsp_insts, n_tsps_per_route)
                objs = torch.tensor(objs, device=device)
            elif subtsp_solver == "lkh":
                costs, tours = lkh_solve(self.opts, tsp_insts, batch_idx)
                costs = get_total_cost(np.array(costs), n_tsps_per_route)
                objs = torch.tensor(costs, device=device)
                # if return_additional_info:
                #     tours_index = np.array(tours)
                #     tours_coords = []
                #     for cord, ri in zip(tsp_insts, tours_index, strict = True):
                #         tours_coords.append(cord[ri])
                #     tours_list.append(np.stack(tours_coords, axis=0))
                del tours
            else:
                tsp_insts = torch.tensor(tsp_insts, device=device)
                tours, objs = glop_eval(tsp_insts, n_tsps_per_route, self.opts)
                # if return_additional_info:
                #     tours_list.append(tours.cpu().numpy())
                del tours
            reward_list.append(objs)

        reward = -torch.stack(reward_list, dim=0)

        return {"reward": reward}

    @torch.no_grad()
    def partition_glop(self, td: TensorDict, actions: torch.Tensor, phase: str):
        """Partition based on the partition actions, from original GLOP
        Args:
            td [bs]: NOTE: different with our partition, this doesn't to be sampled
            actions [bs*n_samples, seq_len]
        Returns:
            tsp_insts_list [bs]: list of tsp instances, each has the size of [sum_num_tsps_of_samples, max_tsp_len, 2]
            n_tsps_per_route_list [bs[n_samples]]: list of number of tsps per route, each element is a list[int]
        """
        batch_size = td.batch_size[0]
        tsp_insts_list = []
        n_tsps_per_route_list = []
        batch_locs = td["locs"].cpu().numpy()
        batch_routes = actions.view(batch_size, self.n_samples, -1).cpu().numpy()
        min_subtsp_scale = 0 if phase == "train" else max(self.opts.revision_lens)

        for coors, routes in zip(batch_locs, batch_routes):
            padded_tsp_pis, n_tsps_per_route = cvrp_to_subtsp(routes, min_subtsp_scale)
            tsp_insts = coors[padded_tsp_pis.astype(int)]
            tsp_insts_list.append(tsp_insts)
            n_tsps_per_route_list.append(n_tsps_per_route)

        return tsp_insts_list, n_tsps_per_route_list
