from rl4co.models.zoo.glop.adapter.tsp_adapter import TSP2SHPPAdapter

try:
    from rl4co.models.zoo.glop.adapter.vrp_adapter import VRP2SubTSPAdapter
except ImportError:
    # In case some dependencies are not installed (e.g., numba)
    VRP2SubTSPAdapter = None


adapter_map = {
    "cvrp": VRP2SubTSPAdapter,
    "cvrpmvc": VRP2SubTSPAdapter,
    "tsp": TSP2SHPPAdapter,
}
