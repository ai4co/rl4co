import math

import torch
import torch.nn as nn


def env_init_embedding(env_name: str, config: dict) -> object:
    return env_embedding(env_name, "init", config)


def env_dynamic_embedding(env_name: str, config: dict) -> object:
    return env_embedding(env_name, "dynamic", config)


def env_embedding(env_name: str, embedding_type: str, config: dict) -> object:
    """Create an embedding object for a given environment name and embedding type.

    Args:
        env_name: The name of the environment.
        config: A dictionary of configuration options for the environment.
        embedding_type: The type of embedding to create, either `init` or `dynamic`.
    """

    embedding_classes = {
        "tsp": {
            "init": TSPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
        "cvrp": {
            "init": VRPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
        "sdvrp": {
            "init": VRPInitEmbedding,
            "dynamic": SDVRPDynamicEmbedding,
        },
        "pctsp": {
            "init": PCTSPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
        "op": {
            "init": OPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
        "dpp": {
            "init": DPPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
        "pdp": {
            "init": PDPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
        "mtsp": {
            "init": MTSPInitEmbedding,
            "dynamic": StaticEmbedding,
        },
    }

    assert embedding_type in ["init", "dynamic"]
    embedding_class = embedding_classes.get(env_name, {}).get(embedding_type, None)

    if embedding_class is None:
        raise ValueError(f"Unknown environment name '{env_name}'")

    return embedding_class(**config)


class TSPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim)

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out


class VRPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(VRPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, demand
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, td):
        # [batch, 1, 2]-> [batch, 1, embedding_dim]
        depot_embedding = self.init_embed_depot(td["depot"])[:, None, :]
        # [batch, n_customer, 2, batch, n_customer, 1]  -> [batch, n_customer, embedding_dim]
        node_embeddings = self.init_embed(
            torch.cat((td["locs"], td["demand"][:, :, None]), -1)
        )
        # [batch, n_customer+1, embedding_dim]
        out = torch.cat((depot_embedding, node_embeddings[..., 1:, :]), 1)
        return out


class PCTSPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PCTSPInitEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, td):  # dict of 'loc', 'deterministic_prize', 'penalty', 'depot'
        # batch, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(td["depot"])[:, None, :]
        # [batch, n_customer, 2, batch, n_customer, 1, batch, n_customer, 1]  -> batch, n_customer, embedding_dim
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    td["observation"][..., 1:, :],
                    td["prize"][..., 1:, None],
                    td["penalty"][..., 1:, None],
                ),
                -1,
            )
        )
        # batch, n_customer+1, embedding_dim
        out = torch.cat((depot_embedding, node_embeddings), 1)
        return out


class OPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(OPInitEmbedding, self).__init__()
        node_dim = 3  # x, y, prize
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, td):  # dict of 'loc', 'prize', 'depot'
        # batch, 1, 2 -> batch, 1, embedding_dim
        depot_embedding = self.init_embed_depot(td["depot"])[:, None, :]
        # [batch, n_customer, 2, batch, n_customer, 1, batch, n_customer, 1]  -> batch, n_customer, embedding_dim
        node_embeddings = self.init_embed(
            torch.cat((td["observation"], td["prize"][:, :, None]), -1)
        )
        # batch, n_customer+1, embedding_dim
        out = torch.cat((depot_embedding, node_embeddings[..., 1:, :]), 1)
        return out


class DPPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(DPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim // 2)  # locs
        self.init_embed_probe = nn.Linear(1, embedding_dim // 2)  # probe

    def forward(self, td):
        node_embeddings = self.init_embed(td["locs"])
        probe_embedding = self.init_embed_probe(
            self._distance_probe(td["locs"], td["probe"])
        )
        return torch.cat([node_embeddings, probe_embedding], -1)

    def _distance_probe(self, locs, probe):
        # Euclidean distance from probe to all locations
        probe_loc = torch.gather(locs, 1, probe.unsqueeze(-1).expand(-1, -1, 2))
        return torch.norm(locs - probe_loc, dim=-1).unsqueeze(-1)


class PDPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embedding_dim)
        self.init_embed_pick = nn.Linear(node_dim * 2, embedding_dim)
        self.init_embed_delivery = nn.Linear(node_dim, embedding_dim)

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)


class MTSPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        """NOTE: new made by Fede. May need to be checked
        """
        super(MTSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, td):
        # embeddings: [batch, n_cities + 1, embedding_dim]
        depot_embedding = self.init_embed_depot(td["locs"][..., 0:1, :])
        node_embedding = self.init_embed(td["locs"][..., 1:, :])
        return torch.cat([depot_embedding, node_embedding], -2)

# class MTSPDynamicEmbedding(nn.Module):
#     def __init__(self, embedding_dim):
#         """NOTE: new made by Fede. May need to be checked
#         We use the current agent idx, the current length of the tour and the max subtour length
#         to modify key, value and logit of multi-head attention
#         """
#         super(MTSPDynamicEmbedding, self).__init__()
#         proj_in_dim = 3  # remaining_agents, current_length, max_subtour_length
#         self.projection = nn.Linear(proj_in_dim, 3 * embedding_dim, bias=False)

#     def forward(self, td):
#         #  [batch, 3]
#         remaining_agents = td["num_agents"] - td["agent_idx"]
#         feats = torch.stack([remaining_agents, td["current_length"], td["max_subtour_length"]], dim=-1)
#         glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
#             feats[..., None, :]
#         ).chunk(3, dim=-1)
#         def _expand(x):
#             return x.expand(-1, td['locs'].size(-2), -1)
#         out = _expand(glimpse_key_dynamic), _expand(glimpse_val_dynamic), _expand(logit_key_dynamic)
#         return out


class SDVRPDynamicEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embedding_dim, bias=False)

    def forward(self, td):
        demands_with_depot = td["demand"][..., :, None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class StaticEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0
