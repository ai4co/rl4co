from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrl.envs import EnvBase

from rl4co.utils.ops import gather_by_index


def env_context(env: Union[str, EnvBase], config: dict) -> object:
    """Get context object for given environment name"""
    context_classes = {
        "tsp": TSPContext,
        "cvrp": VRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
        "pdp": PDPContext,
        "mtsp": MTSPContext,
    }

    env_name = env if isinstance(env, str) else env.name
    context_class = context_classes.get(env_name, None)

    if context_class is None:
        raise ValueError(f"Unknown environment name '{env_name}'")

    return context_class(**config)


class EnvContext(nn.Module):
    def __init__(self, embedding_dim, step_context_dim=None):
        """Get environment context and project it to embedding space"""
        super(EnvContext, self).__init__()
        self.embedding_dim = embedding_dim
        step_context_dim = (
            step_context_dim if step_context_dim is not None else embedding_dim
        )
        self.project_context = nn.Linear(step_context_dim, embedding_dim, bias=False)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class TSPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(TSPContext, self).__init__(embedding_dim, 2 * embedding_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embedding_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (-1,) if td["first_node"].dim() == 1 else (embeddings.size(1), -1)  
        if td["i"][(0,) * td["i"].dim()].item() < 1: # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(batch_size, -1),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(VRPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _state_embedding(self, embeddings, td):
        # TODO: check compatibility between CVRP and SDVRP
        state_embedding = td["capacity"] + td["demand"][..., :1]
        return state_embedding


class PCTSPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(PCTSPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["prize_require"] - td["prize_collect"]
        return state_embedding


class OPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(OPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["length_capacity"]
        return state_embedding


class DPPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(DPPContext, self).__init__(embedding_dim, 2 * embedding_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embedding_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (-1,) if td["first_node"].dim() == 1 else (embeddings.size(1), -1)  
        if td["i"][(0,) * td["i"].dim()].item() < 1: # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings, torch.stack([td["first_node"], td["current_node"]], -1).view(batch_size, -1),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class PDPContext(EnvContext):
    def __init__(self, embedding_dim):
        """From https://arxiv.org/abs/2110.02634"""
        super(PDPContext, self).__init__(embedding_dim, embedding_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class MTSPContext(EnvContext):
    """NOTE: new made by Fede in free style. May need to be checked
    We use as features:
        1. remaining number of agents
        2. the current length of the tour
        3. the max subtour length so far
        4. the current distance from the depot
    """
    def __init__(self, embedding_dim):
        super(MTSPContext, self).__init__(embedding_dim, 2 * embedding_dim)
        proj_in_dim = 4  # remaining_agents, current_length, max_subtour_length, distance_from_depot
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embedding_dim)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding.squeeze()

    def _state_embedding(self, embeddings, td):
        dynamic_feats = torch.stack(
            [
                (td["num_agents"] - td["agent_idx"]).float(),
                td["current_length"],
                td["max_subtour_length"],
                self._distance_from_depot(td),
            ],
            dim=-1,
        )
        return self.proj_dynamic_feats(dynamic_feats)

    def _distance_from_depot(self, td):
        # Euclidean distance from the depot (loc[..., 0, :])
        cur_loc = gather_by_index(td["locs"], td["current_node"])
        return torch.norm(cur_loc - td["locs"][..., 0, :], dim=-1)
