import numpy as np
import torch
import numba as nb

@nb.njit(nogil=True, parallel=False)
def cvrp_to_subtsp(routes, min_reviser_size=0):
    tsp_pis = []
    n_tsps_per_route = []
    max_tsp_len = min_reviser_size
    for route in routes:
        start = 0
        sub_route_count = 0
        for idx, node in enumerate(route[1:], 1):
            if node == 0:
                if route[idx-1] != 0:
                    tsp_pis.append(route[start: idx])
                    thislen = idx-start
                    if thislen>max_tsp_len:
                        max_tsp_len = thislen
                    sub_route_count += 1
                start = idx
        else:
            if node != 0: # handle final routes
                tsp_pis.append(route[start: ])
                thislen = len(route)-start
                if thislen>max_tsp_len:
                    max_tsp_len = thislen
                sub_route_count += 1
        
        n_tsps_per_route.append(sub_route_count)
    padded_tsp_pis = np.zeros((len(tsp_pis), max_tsp_len+1))
    for index, pi in enumerate(tsp_pis):
        padded_tsp_pis[index, :len(pi)] = pi
    return padded_tsp_pis, n_tsps_per_route

@nb.njit()
def get_total_cost(costs, n_tsps_per_route):
    # assert len(costs) == sum(n_tsps_per_route)
    ret = []
    start = 0
    for n in n_tsps_per_route:
        ret.append(costs[start: start+n].sum())
        start += n
    return ret
