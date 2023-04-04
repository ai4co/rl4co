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
    }

    context_class = context_classes.get(env_name, None)

    if context_class is None:
        raise ValueError(f"Unknown environment name '{env_name}'")
    
    return context_class(**config)


class EnvContext(nn.Module):

    def __init__(self, context_dim):
        super(EnvContext, self).__init__()
        self.context_dim = context_dim

    def _prev_node_embedding(self, embeddings, td):
        # current_node = td #state.get_current_node()
        prev_node_embedding = gather_by_index(embeddings, td['current_node'])
        return prev_node_embedding

    def _state_embedding(self, embeddings, td):
        raise NotImplementedError("Implement the embedding for your environment")

    def forward(self, embeddings, td):
        prev_node_embedding = self._prev_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        # Embedding of previous node + remaining capacity
        context_embedding = torch.cat((prev_node_embedding, state_embedding), -1)
        return context_embedding


class TSPContext(EnvContext):
    def __init__(self, context_dim):
        super(TSPContext, self).__init__(context_dim)
        self.W_placeholder = nn.Parameter(torch.Tensor(self.context_dim).uniform_(-1, 1))

    def _state_embedding(self, embeddings, td):
        # first_node = state.first_a
        state_embedding = gather_by_index(embeddings, td["first_node"])
        return state_embedding
        
    def forward(self, embeddings, td):
        # TODO: check whether the vectorized version is correct and FASTER (maybe?)
        batch_size = embeddings.size(0)
        # TODO: vectorize, the following is not vectorized and a bit naive
        # It supposes that the nodes have the same state
        if td['i'][0].item() == 0:
            context_embedding = self.W_placeholder[None, None, :].expand(
                batch_size, 1, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings, torch.cat([td['first_node'], td['current_node']], -1)
            ).view(batch_size, 1, -1)
        return context_embedding


class VRPContext(EnvContext):

    def __init__(self, context_dim):
        super(VRPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["params"]["vehicle_capacity"] - td["used_capacity"][:, :, None]
        return state_embedding


class PCTSPContext(EnvContext):

    def __init__(self, context_dim):
        super(PCTSPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, td):
        state_embedding = td['remaining_prize_to_collect'][:, :, None]
        return state_embedding


class OPContext(EnvContext):

    def __init__(self, context_dim):
        super(OPContext, self).__init__(context_dim)

    def _state_embedding(self, embeddings, td):
        state_embedding = td["remaining_length"][:, :, None]
        return state_embedding

def gather_by_index(source, index):
    target = torch.gather(source, 1, index.unsqueeze(-1).expand(-1, -1, source.size(-1)))
    return target