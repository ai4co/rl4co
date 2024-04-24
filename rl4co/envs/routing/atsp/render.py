import torch

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    try:
        import networkx as nx
    except ImportError:
        log.warn(
            "Networkx is required to visualize the ATSP solution. \
                Please install it with `pip install networkx`"
        )
        return

    td = td.detach().cpu()
    if actions is None:
        actions = td.get("action", None)

    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    src_nodes = actions
    tgt_nodes = torch.roll(actions, 1, dims=0)

    # Plot with networkx
    G = nx.DiGraph(td["cost_matrix"].numpy())
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=800,
        edge_color="white",
    )

    # draw edges src_nodes -> tgt_nodes
    edgelist = [
        (src_nodes[i].item(), tgt_nodes[i].item()) for i in range(len(src_nodes))
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=edgelist, width=2, alpha=1, edge_color="black"
    )
