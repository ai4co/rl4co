import torch
import torch.nn as nn


def env_context(env_name: str, config: dict) -> object:
    """
    Get context for a given environment name and initialize the context object.
    """
    context_classes = {
        "tsp": TSPContext,
        "cvrp": VRPContext,
        "sdvrp": VRPContext,
        "pctsp": PCTSPContext,
        "op": OPContext,
        "dpp": DPPContext,
    }

    context_class = context_classes.get(env_name, None)

    if context_class is None:
        raise ValueError(f"Unknown environment name '{env_name}'")

    return context_class(**config)


class EnvContext(nn.Module):
    def __init__(self, embedding_dim):
        """
        Gather the context for each specific environment and projects it to embedding space
        """
        super(EnvContext, self).__init__()
        self.embedding_dim = embedding_dim
        self.project_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)

    def _prev_node_embedding(self, embeddings, td):
        # current_node = td #state.get_current_node()
        prev_node_embedding = gather_by_index(embeddings, td["current_node"])
        return prev_node_embedding

    def _state_embedding(self, embeddings, td):
        raise NotImplementedError("To implement for each environment")

    def forward(self, embeddings, td):
        prev_node_embedding = self._prev_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat((prev_node_embedding, state_embedding), -1)
        return self.project_context(context_embedding)


class TSPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(TSPContext, self).__init__(embedding_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(self.embedding_dim * 2).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        if td["i"][0].item() == 0:
            context_embedding = self.W_placeholder[None, None, :].expand(
                batch_size, 1, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings, torch.cat([td["first_node"], td["current_node"]], -1)
            ).view(batch_size, 1, -1)
        return self.project_context(context_embedding)


class VRPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(VRPContext, self).__init__(embedding_dim)

    def _state_embedding(self, embeddings, td):
        state_embedding = (
            td["params"]["vehicle_capacity"] - td["used_capacity"][:, :, None]
        )
        return state_embedding


class PCTSPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(PCTSPContext, self).__init__(embedding_dim)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["remaining_prize_to_collect"][:, :, None]
        return state_embedding


class OPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(OPContext, self).__init__(embedding_dim)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["remaining_length"][:, :, None]
        return state_embedding


class DPPContext(EnvContext):
    def __init__(self, embedding_dim):
        super(DPPContext, self).__init__(embedding_dim)
        self.W_placeholder = nn.Parameter(torch.Tensor(embedding_dim).uniform_(-1, 1))
        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        if td["i"][0].item() == 0:
            context_embedding = self.W_placeholder[None, None, :].expand(
                batch_size, 1, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(embeddings, td["current_node"]).view(
                batch_size, 1, -1
            )
        return self.project_context(context_embedding)


@torch.jit.script
def gather_by_index(source, index):
    target = torch.gather(
        source, 1, index.unsqueeze(-1).expand(-1, -1, source.size(-1))
    )
    return target
