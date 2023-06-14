from numpy import fmax
import torch
from typing import Optional
from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.data.utils import load_npz_to_tensordict
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
CAPACITIES = {
        10: 20.0,
        15: 25.0,
        20: 30.0,
        30: 33.0,
        40: 37.0,
        50: 40.0,
        60: 43.0,
        75: 45.0,
        100: 50.0,
        125: 55.0,
        150: 60.0,
        200: 70.0,
        500: 100.0,
        1000: 150.0,
    }


class CVRPEnv(RL4COEnvBase):
    """Capacitated Vehicle Routing Problem (CVRP) environment.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is the -infinite unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Args:
        num_loc (int): number of locations (cities) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc (float): minimum value for the location coordinates
        max_loc (float): maximum value for the location coordinates
        min_demand (float): minimum value for the demand of each customer
        max_demand (float): maximum value for the demand of each customer
        vehicle_capacity (float): capacity of the vehicle
        capacity (float): capacity of the vehicle
        td_params (TensorDict): parameters of the environment
    """

    name = "cvrp"

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0,
        max_loc: float = 1,
        min_demand: float = 1,
        max_demand: float = 10,
        vehicle_capacity: float = 1.0,
        capacity: float = None,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.capacity = CAPACITIES.get(num_loc, None) if capacity is None else capacity
        if self.capacity is None:
            raise ValueError(
                f"Capacity for {num_loc} locations is not defined. Please provide a capacity manually."
            )
        self.vehicle_capacity = vehicle_capacity
        self._make_spec(td_params)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        # Update the state
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (
            current_node != 0
        ).float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # SECTION: get mask
        visited_loc = visited[..., 1:]

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"][:, None, :] + used_capacity[..., None] > 1.0

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (current_node == 0) & ((mask_loc == 0).int().sum(-1) > 0)

        # Action mask will be inverse of unfeasible actions
        feasible_actions = ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.ones_like(done) * float("-inf")

        return TensorDict(
            {
                "next": {
                    "locs": td["locs"],
                    "demand": td["demand"],
                    "current_node": current_node,
                    "used_capacity": used_capacity,
                    "vehicle_capacity": td["vehicle_capacity"],
                    "visited": visited,
                    "action_mask": feasible_actions,
                    "reward": reward,
                    "done": done,
                }
            },
            td.shape,
        )

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        if batch_size is None:
            batch_size = self.batch_size if td is None else td["locs"].shape[:-2]

        if td is None or td.is_empty():
            td = self.generate_data(batch_size=batch_size)

        self.device = td.device

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)
        current_node = torch.zeros(*batch_size, 1, dtype=torch.long, device=self.device)
        used_capacity = torch.zeros((*batch_size, 1), device=self.device)

        _, n_loc, _ = td["locs"].size()
        visited = torch.zeros(
            (*batch_size, 1, n_loc + 1), dtype=torch.uint8, device=self.device
        )

        # SECTION: get mask
        visited_loc = visited[..., 1:]

        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"][:, None, :] + used_capacity[..., None] > 1.0

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (current_node == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        feasible_actions = ~torch.cat((mask_depot[..., None], mask_loc), -1).squeeze(-2)

        # Vehicle capacity as a feature
        vehicle_capacity = torch.full(
            (*batch_size, 1), self.vehicle_capacity, device=self.device
        )

        return TensorDict(
            {
                "locs": locs,
                "demand": td["demand"],
                "current_node": current_node,
                "used_capacity": used_capacity,
                "vehicle_capacity": vehicle_capacity,
                "visited": visited,
                "action_mask": feasible_actions,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, td_params: TensorDict):
        """Make the observation and action specs from the parameters."""
        self.observation_spec = CompositeSpec(
            locs=BoundedTensorSpec(
                minimum=self.min_loc,
                maximum=self.max_loc,
                shape=(self.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=BoundedTensorSpec(
                minimum=-self.capacity,
                maximum=self.max_demand,
                shape=(self.num_loc, 1), # demand is only for customers
                dtype=torch.float32,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.input_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_loc + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    @staticmethod
    def get_reward(td, actions) -> TensorDict:
        # Check if tour is valid, i.e. contain 0 to n-1
        batch_size, graph_size = td['demand'].size()
        sorted_pi = actions.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=sorted_pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                -td['vehicle_capacity'],
                td['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, actions)

        used_cap = torch.zeros_like(td['demand'][:, 0])
        for i in range(actions.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= td['vehicle_capacity'] + 1e-5).all(), "Used more than capacity"

        # Calculate the reward: - distance from all locations. 
        # We add the depot as first so we calculate also depot-first and depot-last tours with roll
        depot = td["locs"][..., 0:1, :]
        loc_gathered = torch.cat([depot, gather_by_index(td["locs"], actions)], dim=1)
        loc_gathered_next = torch.roll(loc_gathered, 1, dims=1)
        return -((loc_gathered_next - loc_gathered).norm(p=2, dim=2).sum(1))

    def generate_data(self, batch_size) -> TensorDict:

        # Batch size input check
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        # Initialize the locations (including the depot which is always the first node)
        locs_with_depot = (
            torch.FloatTensor(*batch_size, self.num_loc + 1, 2)
            .uniform_(self.min_loc, self.max_loc)
            .to(self.device)
        )

        # Initialize the demand for nodes except the depot
        # Demand sampling Following Kool et al. (2019)
        # Generates a slightly different distribution than using torch.randint
        demand = (
            (
                torch.FloatTensor(*batch_size, self.num_loc)
                .uniform_(self.min_demand - 1, self.max_demand - 1)
                .int()
                + 1
            )
            .float()
            .to(self.device)
        )

        # Support for heterogeneous capacity if provided
        if not isinstance(self.capacity, torch.Tensor):
            capacity = torch.full((*batch_size,), self.capacity, device=self.device)
        else:
            capacity = self.capacity

        return TensorDict(
            {
                "locs": locs_with_depot[..., 1:, :],
                "depot": locs_with_depot[..., 0, :],
                "demand": demand / CAPACITIES[self.num_loc],
                "capacity": capacity,
            },
            batch_size=batch_size,
        )

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        td_load = load_npz_to_tensordict(fpath)
        td_load.set_("demand", td_load["demand"] / td_load["capacity"][:, None])
        return td_load

    @staticmethod
    def render(td: TensorDict, actions=None, ax=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm, colormaps

        base = colormaps['rainbow']
        color_list = base(np.linspace(0, 1, 6))
        cmap_name = base.name + str(6)
        out = base.from_list(cmap_name, color_list, 6)

        if ax is None:
            # Create a plot of the nodes
            _, ax = plt.subplots()

        td = td.detach().cpu()
        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]

        locs = td["locs"]
        demands = td["demand"]

        if actions is None:
            actions = td.get("action", None)

        # gather locs in order of action if available
        if actions is None:
            log.warning("No action in TensorDict, rendering unsorted locs")
        else:
            locs = locs

        # Cat the first node to the end to complete the tour
        x, y = locs[:, 0], locs[:, 1]

        # plot depot
        ax.scatter(
            locs[0, 0], locs[0, 1], 
            edgecolors=cm.Set2(2),
            facecolors='none',
            s=100,
            linewidths=2,
            marker='s',
            alpha=1,
        )

        # plot visited nodes
        ax.scatter(
            x[1:], y[1:],
            edgecolors=cm.Set2(0),
            facecolors='none',
            s=100,
            linewidths=2,
            marker='o',
            alpha=1,
        )

        print(len(locs))
        print(len(demands))
        # plot demand bars
        for node_idx in range(1, len(locs)):
            ax.add_patch(
                plt.Rectangle(
                    (locs[node_idx,0]-0.005, locs[node_idx,1]+0.025), 
                    0.01, 
                    demands[node_idx-1]*0.1,
                    edgecolor=cm.Set2(0),
                    facecolor=cm.Set2(0),
                    fill=True,
                )
            )

        # text demand
        for node_idx in range(1, len(locs)):
            ax.text(
                locs[node_idx, 0], locs[node_idx, 1]-0.025, 
                f'{demands[node_idx-1].item():.2f}',
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=10,
                color=cm.Set2(0),
            )

        # text depot
        ax.text(
            locs[0, 0], locs[0, 1]-0.025, 
            f'Depot',
            horizontalalignment='center',
            verticalalignment='top',
            fontsize=10,
            color=cm.Set2(2),
        )

        # plot actions
        color_idx = 0
        for action_idx in range(len(actions)-1):
            if actions[action_idx] == 0:
                color_idx += 1
            from_loc = locs[actions[action_idx]]
            to_loc = locs[actions[action_idx+1]]
            ax.plot(
                [from_loc[0], to_loc[0]], [from_loc[1], to_loc[1]],
                color=out(color_idx),
            )
            ax.annotate(
                "", 
                xy=(to_loc[0], to_loc[1]),
                xytext=(from_loc[0], from_loc[1]),
                arrowprops=dict(arrowstyle="->", color=out(color_idx)),
                annotation_clip=False,
            )

        # setup
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='both', color='black', ls='--', alpha=0.1)

        # Set plot title and axis labels
        title_font = {'fontsize': 14}
        label_font = {'fontsize': 12}
        ax.set_title("CVRP Solution", fontdict=title_font)
        ax.set_xlabel("X Coordinate", fontdict=label_font)
        ax.set_ylabel("Y Coordinate", fontdict=label_font)

        # plt.tight_layout()
        plt.show()



