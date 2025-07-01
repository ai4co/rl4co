import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import cm, colormaps

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    num_routine = (actions == 0).sum().item() + 2
    base = colormaps["nipy_spectral"]
    color_list = base(np.linspace(0, 1, num_routine))
    cmap_name = base.name + str(num_routine)
    out = base.from_list(cmap_name, color_list, num_routine)

    if ax is None:
        # Create a plot of the nodes
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)

    # if batch_size greater than 0 , we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        actions = actions[0]

    locs = td["locs"]
    scale_demand = td["capacity"][0]
    demands = td["demand"] * scale_demand

    # add the depot at the first action and the end action
    actions = torch.cat([torch.tensor([0]), actions, torch.tensor([0])])

    # gather locs in order of action if available
    if actions is None:
        log.warning("No action in TensorDict, rendering unsorted locs")
    else:
        locs = locs

    # Cat the first node to the end to complete the tour
    x, y = locs[:, 0], locs[:, 1]

    # plot depot
    ax.scatter(
        locs[0, 0],
        locs[0, 1],
        edgecolors=cm.Set2(2),
        facecolors="none",
        s=100,
        linewidths=2,
        marker="s",
        alpha=1,
    )

    # plot visited nodes
    ax.scatter(
        x[1:],
        y[1:],
        edgecolors=cm.Set2(0),
        facecolors="none",
        s=50,
        linewidths=2,
        marker="o",
        alpha=1,
    )

    # plot demand bars
    for node_idx in range(1, len(locs)):
        ax.add_patch(
            plt.Rectangle(
                (locs[node_idx, 0] - 0.005, locs[node_idx, 1] + 0.015),
                0.01,
                demands[node_idx - 1] / (scale_demand * 10),
                edgecolor=cm.Set2(0),
                facecolor=cm.Set2(0),
                fill=True,
            )
        )

    # text demand
    for node_idx in range(1, len(locs)):
        ax.text(
            locs[node_idx, 0],
            locs[node_idx, 1] - 0.025,
            f"{demands[node_idx-1].item():.2f}",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color=cm.Set2(0),
        )

    # text depot
    ax.text(
        locs[0, 0],
        locs[0, 1] - 0.025,
        "Depot",
        horizontalalignment="center",
        verticalalignment="top",
        fontsize=10,
        color=cm.Set2(2),
    )

    # plot actions
    color_idx = 0
    for action_idx in range(len(actions) - 1):
        if actions[action_idx] == 0:
            color_idx += 1
        from_loc = locs[actions[action_idx]]
        to_loc = locs[actions[action_idx + 1]]
        ax.plot(
            [from_loc[0], to_loc[0]],
            [from_loc[1], to_loc[1]],
            color=out(color_idx),
            lw=1,
        )
        ax.annotate(
            "",
            xy=(to_loc[0], to_loc[1]),
            xytext=(from_loc[0], from_loc[1]),
            arrowprops=dict(arrowstyle="-|>", color=out(color_idx)),
            size=15,
            annotation_clip=False,
        )

    return ax
