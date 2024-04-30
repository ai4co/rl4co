from typing import Union, Callable

import torch

from torch.distributions import Uniform
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import get_sampler, Generator

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


class CVRPGenerator(Generator):
    """Data generator for the Capacitated Vehicle Routing Problem (CVRP).
    Args:
        num_loc: number of locations (cities) in the VRP, without the depot. (e.g. 10 means 10 locs + 1 depot)
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        loc_distribution: distribution for the location coordinates
        min_demand: minimum value for the demand of each customer
        max_demand: maximum value for the demand of each customer
        demand_distribution: distribution for the demand of each customer
        capacity: capacity of the vehicle

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc + 1, 2]: locations of each customer and the depot
            depot [batch_size, 2]: location of the depot
            demand [batch_size, num_loc + 1]: demand of each customer and the depot, 
                while the demand of the depot is a placeholder
            capacity [batch_size]: capacity of the vehicle
    """
    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        depot_distribution: Union[
            int, float, str, type, Callable
        ] = Uniform,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution: Union[
            int, float, type, Callable
        ] = Uniform,
        vehicle_capacity: float = 1.0,
        capacity: float = None,
        **kwargs
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_capacity = vehicle_capacity

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler("loc", loc_distribution, min_loc, max_loc, **kwargs)

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = get_sampler("depot", depot_distribution, min_loc, max_loc, **kwargs)

        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler("demand", demand_distribution, min_demand-1, max_demand-1, **kwargs)
        
        # Capacity
        self.capacity = kwargs.get("capacity", None)
        if self.capacity is None: # If not provided, use the default capacity from Kool et al. 2019
            self.capacity = CAPACITIES.get(num_loc, None)
        if self.capacity is None: # If not in the table keys, find the cloest number of nodes as the key
            cloest_num_loc = min(CAPACITIES.keys(), key=lambda x: abs(x - num_loc))
            self.capacity = CAPACITIES[cloest_num_loc]
            log.warning(
                f"The capacity capacity for {num_loc} locations is not defined. Using the closest capacity: {self.capacity}\
                    with {cloest_num_loc} locations."
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample depot
        depot = self.depot_sampler.sample((*batch_size, 2))
        
        # Sample demands
        demand = self.demand_sampler.sample((*batch_size, self.num_loc))
        demand = (demand.int() + 1).float()

        # Sample capacities
        capacity = torch.full((*batch_size, 1), self.capacity)

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demand": demand / self.capacity,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )
