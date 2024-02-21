import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index


def env_context_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment context embedding. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Usually consists of a projection of gathered node embeddings and features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPContext,
        "atsp": TSPContext,
        "cvrp": VRPContext,
        "cvrptw": VRPTWContext,
        "svrp": SVRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "spctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
        "mdpp": DPPContext,
        "pdp": PDPContext,
        "mtsp": MTSPContext,
        "smtwtp": SMTWTPContext,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available context embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class EnvContext(nn.Module):
    """Base class for environment context embeddings. The context embedding is used to modify the
    query embedding of the problem node of the current partial solution.
    Consists of a linear layer that projects the node features to the embedding space."""

    def __init__(self, embedding_dim, step_context_dim=None, linear_bias=False):
        super(EnvContext, self).__init__()
        self.embedding_dim = embedding_dim
        step_context_dim = (
            step_context_dim if step_context_dim is not None else embedding_dim
        )
        self.project_context = nn.Linear(
            step_context_dim, embedding_dim, bias=linear_bias
        )

    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        """Get state embedding"""
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class TSPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embedding_dim):
        super(TSPContext, self).__init__(embedding_dim, 2 * embedding_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * self.embedding_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embedding_dim):
        super(VRPContext, self).__init__(
            embedding_dim=embedding_dim, step_context_dim=embedding_dim + 1
        )

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding


class VRPTWContext(VRPContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
        - current time
    """

    def __init__(self, embedding_dim):
        super(VRPContext, self).__init__(
            embedding_dim=embedding_dim, step_context_dim=embedding_dim + 2
        )

    def _cur_node_embedding(self, embeddings, td):
        return super()._cur_node_embedding(embeddings, td).reshape(embeddings.size(0), -1)

    def _state_embedding(self, embeddings, td):
        capacity = super()._state_embedding(embeddings, td)
        current_time = td["current_time"]
        return torch.cat([capacity, current_time], -1)


class SVRPContext(EnvContext):
    """Context embedding for the Skill Vehicle Routing Problem (SVRP).
    Project the following to the embedding space:
        - current node embedding
        - current technician
    """

    def __init__(self, embedding_dim):
        super(SVRPContext, self).__init__(
            embedding_dim=embedding_dim, step_context_dim=embedding_dim
        )

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class PCTSPContext(EnvContext):
    """Context embedding for the Prize Collecting TSP (PCTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining prize (prize_required - cur_total_prize)
    """

    def __init__(self, embedding_dim):
        super(PCTSPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = torch.clamp(
            td["prize_required"] - td["cur_total_prize"], min=0
        )[..., None]
        return state_embedding


class OPContext(EnvContext):
    """Context embedding for the Orienteering Problem (OP).
    Project the following to the embedding space:
        - current node embedding
        - remaining distance (max_length - tour_length)
    """

    def __init__(self, embedding_dim):
        super(OPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["max_length"][..., 0] - td["tour_length"]
        return state_embedding[..., None]


class DPPContext(EnvContext):
    """Context embedding for the Decap Placement Problem (DPP), EDA (electronic design automation).
    Project the following to the embedding space:
        - current cell embedding
    """

    def __init__(self, embedding_dim):
        super(DPPContext, self).__init__(embedding_dim)

    def forward(self, embeddings, td):
        """Context cannot be defined by a single node embedding for DPP, hence 0.
        We modify the dynamic embedding instead to capture placed items
        """
        return embeddings.new_zeros(embeddings.size(0), self.embedding_dim)


class PDPContext(EnvContext):
    """Context embedding for the Pickup and Delivery Problem (PDP).
    Project the following to the embedding space:
        - current node embedding
    """

    def __init__(self, embedding_dim):
        super(PDPContext, self).__init__(embedding_dim, embedding_dim)

    def forward(self, embeddings, td):
        cur_node_embedding = self._cur_node_embedding(embeddings, td).squeeze()
        return self.project_context(cur_node_embedding)


class MTSPContext(EnvContext):
    """Context embedding for the Multiple Traveling Salesman Problem (mTSP).
    Project the following to the embedding space:
        - current node embedding
        - remaining_agents
        - current_length
        - max_subtour_length
        - distance_from_depot
    """

    def __init__(self, embedding_dim, linear_bias=False):
        super(MTSPContext, self).__init__(embedding_dim, 2 * embedding_dim)
        proj_in_dim = (
            4  # remaining_agents, current_length, max_subtour_length, distance_from_depot
        )
        self.proj_dynamic_feats = nn.Linear(proj_in_dim, embedding_dim, bias=linear_bias)

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


class SMTWTPContext(EnvContext):
    """Context embedding for the Single Machine Total Weighted Tardiness Problem (SMTWTP).
    Project the following to the embedding space:
        - current node embedding
        - current time
    """

    def __init__(self, embedding_dim):
        super(SMTWTPContext, self).__init__(embedding_dim, embedding_dim + 1)

    def _cur_node_embedding(self, embeddings, td):
        cur_node_embedding = gather_by_index(embeddings, td["current_job"])
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["current_time"]
        return state_embedding
