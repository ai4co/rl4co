import torch

from tensordict.tensordict import TensorDict


def render(td: TensorDict, actions=None, ax=None):
    import matplotlib.pyplot as plt

    markersize = 8

    td = td.detach().cpu()

    # If batch_size greater than 0, we need to select the first batch element
    if td.batch_size != torch.Size([]):
        td = td[0]
        if actions is not None:
            actions = actions[0]

    n_depots = td["capacity"].shape[-1]
    n_pickups = (td["locs"].size(-2) - n_depots) // 2

    # Variables
    init_deliveries = td["to_deliver"][n_depots:]
    delivery_locs = td["locs"][n_depots:][~init_deliveries.bool()]
    pickup_locs = td["locs"][n_depots:][init_deliveries.bool()]
    depot_locs = td["locs"][:n_depots]
    actions = actions if actions is not None else td["action"]

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Split actions into tours (segments starting from depot)
    tours = []
    current_tour = []
    current_depot = actions[0]  # Start from first depot

    for action in actions:
        if action < n_depots:
            if current_tour:  # If we have a non-empty tour
                # Add the current tour with its starting depot
                tours.append((current_depot, current_tour))
            current_depot = action
            current_tour = []
        else:
            current_tour.append(action)

    # Add the last tour if it exists
    if current_tour:
        tours.append((current_depot, current_tour))

    # Plot each tour with a different color
    for tour_idx, (depot, tour) in enumerate(tours):
        color = f"C{tour_idx}"  # Use matplotlib's color cycle

        # Plot from depot to first location
        if tour:  # Only if tour is not empty
            from_loc = td["locs"][depot]
            to_loc = td["locs"][tour[0]]
            ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color)
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="->", color=color),
                annotation_clip=False,
            )

            # Plot connections between tour locations
            for i in range(len(tour) - 1):
                from_loc = td["locs"][tour[i]]
                to_loc = td["locs"][tour[i + 1]]
                ax.plot([from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]], color=color)
                ax.annotate(
                    "",
                    xy=(to_loc[0], to_loc[1]),
                    xytext=(from_loc[0], from_loc[1]),
                    arrowprops=dict(arrowstyle="->", color=color),
                    annotation_clip=False,
                )

            # Plot return to depot in faint grey dashed line
            from_loc = td["locs"][tour[-1]]
            to_loc = td["locs"][depot]
            ax.plot(
                [from_loc[0], to_loc[0]],
                [from_loc[1], to_loc[1]],
                color="grey",
                linestyle="--",
                alpha=0.3,
            )
            ax.annotate(
                "",
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="->", color="grey", alpha=0.3),
                annotation_clip=False,
            )

    # Annotate node locations
    for i, loc in enumerate(td["locs"]):
        ax.annotate(
            str(i),
            (loc[0], loc[1]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
        )

    # Plot depots
    for i, depot_loc in enumerate(depot_locs):
        ax.plot(
            depot_loc[0],
            depot_loc[1],
            "tab:green",
            marker="s",
            markersize=markersize,
            label="Depot" if i == 0 else None,
        )

    # Plot pickup locations
    for i, pickup_loc in enumerate(pickup_locs):
        ax.plot(
            pickup_loc[0],
            pickup_loc[1],
            "tab:red",
            marker="^",
            markersize=markersize,
            label="Pickup" if i == 0 else None,
        )

    # Plot delivery locations
    for i, delivery_loc in enumerate(delivery_locs):
        ax.plot(
            delivery_loc[0],
            delivery_loc[1],
            "tab:blue",
            marker="x",
            markersize=markersize,
            label="Delivery" if i == 0 else None,
        )

    # Plot pickup and delivery pairs
    for i in range(n_pickups):
        pickup_loc = td["locs"][n_depots + i]
        delivery_loc = td["locs"][n_depots + n_pickups + i]
        ax.plot(
            [pickup_loc[0], delivery_loc[0]],
            [pickup_loc[1], delivery_loc[1]],
            "k--",
            alpha=0.5,
        )

    return ax
