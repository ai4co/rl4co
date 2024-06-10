from dataclasses import dataclass
from typing import Optional

import numpy as np

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tensordict import TensorDict
from torch import Tensor

from . import pyvrp

ORTOOLS_SCALING_FACTOR = 100_000


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves an CVRP instance with OR-Tools.

    Parameters
    ----------
    instance
        The CVRP instance to solve.
    max_runtime
        The maximum runtime for the solver.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.

    Notes
    -----
    This function depends on PyVRP's data converter to convert the CVRP
    instance to an OR-Tools compatible format. Future versions should
    implement a direct conversion.
    """
    data = instance2data(instance)
    action, cost = _solve(data, max_runtime)
    cost /= ORTOOLS_SCALING_FACTOR
    cost *= -1

    return action, cost


@dataclass
class ORToolsData:
    """
    Convenient dataclass for instance data when using OR-Tools as solver.

    Parameters
    ----------
    depot
        The depot index.
    distance_matrix
        The distance matrix between locations.
    vehicle_capacities
        The capacity of each vehicle.
    demands
        The demands of each location.
    """

    depot: int
    distance_matrix: list[list[int]]
    vehicle_capacities: list[int]
    demands: list[int]

    @property
    def num_locations(self) -> int:
        return len(self.distance_matrix)


def instance2data(instance: TensorDict) -> ORToolsData:
    """
    Converts an CVRP instance to an ORToolsData instance.
    """
    # TODO: Do not use PyVRP's data converter.
    data = pyvrp.instance2data(instance, ORTOOLS_SCALING_FACTOR)

    capacities = [
        veh_type.capacity
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]

    demands = [0] + [client.delivery for client in data.clients()]
    distances = data.distance_matrix().copy()

    return ORToolsData(
        depot=0,
        distance_matrix=distances.tolist(),
        vehicle_capacities=capacities,
        demands=demands,
    )


def _solve(data: ORToolsData, max_runtime: float, log: bool = False):
    """
    Solves an instance with OR-Tools.

    Parameters
    ----------
    data
        The instance data.
    max_runtime
        The maximum runtime in seconds.
    log
        Whether to log the search.

    Returns
    -------
    tuple[list[list[int]], int]
        A tuple containing the routes and the objective value.
    """
    # Manager for converting between nodes (location indices) and index
    # (internal CP variable indices).
    manager = pywrapcp.RoutingIndexManager(
        data.num_locations, data.num_vehicles, data.depot
    )
    routing = pywrapcp.RoutingModel(manager)

    # Set arc costs equal to distances.
    distance_transit_idx = routing.RegisterTransitMatrix(data.distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_idx)

    # Vehicle capacity constraint.
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitVector(data.demands),
        0,  # null capacity slack
        data.vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Demand",
    )

    # Setup search parameters.
    params = pywrapcp.DefaultRoutingSearchParameters()

    gls = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.local_search_metaheuristic = gls

    params.time_limit.FromSeconds(int(max_runtime))  # only accepts int
    params.log_search = log

    solution = routing.SolveWithParameters(params)
    action = solution2action(data, manager, routing, solution)
    objective = solution.ObjectiveValue()

    return action, objective


def solution2action(data, manager, routing, solution) -> list[list[int]]:
    """
    Converts an OR-Tools solution to routes.
    """
    routes = []
    distance = 0  # for debugging

    for vehicle_idx in range(data.num_vehicles):
        index = routing.Start(vehicle_idx)
        route = []
        route_cost = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx)

        if clients := route[1:]:  # ignore depot
            routes.append(clients)
            distance += route_cost

    return [visit for route in routes for visit in route + [0]]
