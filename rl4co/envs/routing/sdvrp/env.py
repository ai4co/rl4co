import torch

from tensordict.tensordict import TensorDict

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from ..cvrp.env import CVRPEnv
from ..cvrp.generator import CVRPGenerator

log = get_pylogger(__name__)


class SDVRPEnv(CVRPEnv):
    """Split Delivery Vehicle Routing Problem (SDVRP) environment.
    SDVRP is a generalization of CVRP, where nodes can be visited multiple times and a fraction of the demand can be met.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is the -infinite unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - location of the depot.
        - locations and demand/remaining demand of each customer (city).
        - current location of the vehicle.
        - the remaining capacity of the vehicle.

    Constraints:
        - the tour starts and ends at the depot.
        - each city can be visited multiple times.
        - the vehicle cannot visit cities exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.

    Finish Condition:
        - the vehicle has finished all cities demand and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "sdvrp"

    def __init__(
        self,
        generator: CVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(generator, generator_params, **kwargs)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]

        # Get demand for selected node
        selected_demand = gather_by_index(td["demand"], current_node, squeeze=False)

        # Get available capacity for selected node demand
        delivered_demand = torch.min(selected_demand, td["capacity"] - td["used_capacity"])

        # If not depot, add demand to used capacity; if depot, reset used capacity to 0
        used_capacity = (td["used_capacity"] + delivered_demand) * (current_node != 0).float()

        # Update demand
        demand = td["demand"].scatter_add(-1, current_node, -delivered_demand)

        done = ~(demand > 1e-5).any(-1)
        done = done[:, None]
        reward = torch.zeros_like(done)

        td.update(
            {
                "demand": demand,
                "current_node": current_node,
                "used_capacity": used_capacity,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        action_mask = td["demand"] > 0

        # Can not visit other nodes if the vehicle is full
        action_mask &= td["used_capacity"] <= td["capacity"] - 1e-5

        # Depot is always available except the vehicle is already at the depot while there are still unserved nodes
        action_mask[:, :1] = (td["current_node"] != 0) | td["done"]

        return action_mask

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that the solution is valid (all demand is satisfied)"""
        batch_size, _, _ = td["locs"].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat((-td["capacity"], td["demand"][:, 1:]), 1)

        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(td["demand"][..., 0])
        a_prev = None
        for a in actions.transpose(0, 1):
            assert (
                a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all()
            ), "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], td["capacity"].squeeze(-1) - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"
