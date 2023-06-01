import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrl.envs import EnvBase


def env_init_embedding(env: Union[str, EnvBase], config: dict) -> object:
    return env_embedding(env, "init", config)


def env_dynamic_embedding(env: Union[str, EnvBase], config: dict) -> object:
    return env_embedding(env, "dynamic", config)


def env_embedding(
    env: Union[str, EnvBase], embedding_type: str, config: dict
) -> object:
    """Create an embedding object for a given environment name and embedding type.

    Args:
        env: Environment or its name.
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
        "mdpp": {
            "init": MDPPInitEmbedding,
            "dynamic": DPPDynamicEmbedding,
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

    assert embedding_type in [
        "init",
        "dynamic",
    ], "Unknown embedding type. Must be one of 'init' or 'dynamic'"
    env_name = env if isinstance(env, str) else env.name
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
        depot, customers = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        # [batch, n_customer, 2, batch, n_customer, 1]  -> [batch, n_customer, embedding_dim]
        node_embeddings = self.init_embed(
            torch.cat((customers, td["demand"][..., None]), -1)
        )
        # [batch, n_customer+1, embedding_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
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


class MDPPInitEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(MDPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim // 2)  # locs
        self.init_embed_probes = nn.Linear(node_dim + 1, embedding_dim // 2)  # locs + is_probe

    def forward(self, td):
        node_embeddings = self.init_embed(td["locs"])
        probes_with_locs = torch.cat(
            [td["locs"], td["probe"].float()[...,None]], dim=-1
        )  # [batch, n_locs, 3] # x, y, is_probe
        probes_embedding = self.init_embed_probes(probes_with_locs)
        return torch.cat([node_embeddings, probes_embedding], -1)


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
        """NOTE: new made by Fede. May need to be checked"""
        super(MTSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_depot = nn.Linear(2, embedding_dim)  # depot embedding

    def forward(self, td):
        # embeddings: [batch, n_cities + 1, embedding_dim]
        depot_embedding = self.init_embed_depot(td["locs"][..., 0:1, :])
        node_embedding = self.init_embed(td["locs"][..., 1:, :])
        return torch.cat([depot_embedding, node_embedding], -2)


class SDVRPDynamicEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embedding_dim, bias=False)

    def forward(self, td):
        demands_with_depot = td["demand"][..., None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class DPPDynamicEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(DPPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(2, 3 * embedding_dim, bias=False)

    def forward(self, td):
        unavailable, keepouts, probes = ~td["action_mask"].clone(), td["keepout"].clone(), td["probe"].clone()
        placed_decaps = unavailable  & ~(keepouts | probes)
        decaps_and_probes = torch.stack([placed_decaps.float(), probes.float()], dim=-1)
        
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            decaps_and_probes
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class StaticEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0
