import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CVRPMVCEnv(CVRPEnv):
    """Capacitated Vehicle Routing Problem (CVRP) with maximum vehicle constraint environment."""

    name = "cvrpmvc"

    def _step(self, td: TensorDict) -> TensorDict:
        vehicles_used = td["vehicles_used"] + (
            (td["action"].unsqueeze(-1) == 0) & (td["current_node"] != 0)
        )

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

        demand_remaining = td["demand_remaining"] - selected_demand

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node, 1)

        # SECTION: get done
        done = visited.sum(-1) == visited.size(-1)
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "used_capacity": used_capacity,
                "vehicles_used": vehicles_used,
                "demand_remaining": demand_remaining,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: TensorDict | None = None, batch_size: list | None = None
    ) -> TensorDict:
        td = super()._reset(td, batch_size)
        batch_size = batch_size or list(td.batch_size)
        td.set(
            "vehicles_used",
            torch.ones((*batch_size, 1), dtype=torch.int, device=td.device),
        )
        td.set("demand_remaining", td["demand"].sum(-1, keepdim=True))
        td.set(
            "max_vehicle", torch.ceil(td["demand_remaining"] / td["vehicle_capacity"]) + 1
        )
        return td

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"]

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap

        if "vehicles_used" in td.keys():
            max_vehicle = td["max_vehicle"]
            demand_remaining = td["demand_remaining"]
            capacity_remaining = (max_vehicle - td["vehicles_used"]) * td[
                "vehicle_capacity"
            ]
            mask_depot = (  # mask the depot
                (td["current_node"] == 0)  # if the depot is just visited
                | (
                    demand_remaining > capacity_remaining
                )  # or the unassigned vehicles' capacity can't sastify remaining demands
            ) & ~torch.all(
                mask_loc, dim=-1, keepdim=True
            )  # unless there's no other choices
        else:
            # Cannot visit the depot if just visited and still unserved nodes
            mask_depot = (td["current_node"] == 0) & ~torch.all(
                mask_loc, dim=-1, keepdim=True
            )
        return ~torch.cat((mask_depot, mask_loc), -1)
