import os
import numpy as np
from subprocess import check_output, check_call
import tempfile
import time

from torch import Tensor
from tensordict.tensordict import TensorDict


SCALING_FACTOR = 10_000_000
PRIZE_SCALING_FACTOR = 100
MAX_LENGTH_TOL = 1e-5


def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves the OP instance with Compass.

    Parameters
    ----------
    instance
        The OP instance to solve.
    max_runtime
        Maximum runtime for the solver.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple consisting of the action and the cost, respectively.
    """
    executable = os.path.abspath(os.path.join('rl4co', 'envs', 'routing', 'op', 'baselines', 'compass', 'compass'))
    action, cost = _solve(executable, instance)

    return action, cost


def _solve(executable, td):
    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.oplib")
        tour_filename = os.path.join(tempdir, "output.tour")
        log_filename = os.path.join(tempdir, "problem.log")
        
        depot = td["depot"].tolist()
        loc = td["locs"].tolist()
        prize = td['prize'].tolist()  #(td["prize"] * PRIZE_SCALING_FACTOR).int().tolist()
        max_length = td["max_length"].item()
        
        starttime = time.time()
        write_oplib(problem_filename, depot, loc, prize, max_length)
        
        with open(log_filename, 'w') as f:
            start = time.time()
            check_call([executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename], stdout=f, stderr=f)
            
            duration = time.time() - start
        
        # import pdb; pdb.set_trace()
        # params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        # write_compass_par(param_filename, params)
        tour = read_oplib(tour_filename, n=td["prize"].size(0))
        
        if not calc_op_length(depot, loc, tour) <= max_length:
            print("Warning: length exceeds max length:", calc_op_length(depot, loc, tour), max_length)
        assert calc_op_length(depot, loc, tour) <= max_length + MAX_LENGTH_TOL, "Tour exceeds max_length!"
        
        import pdb; pdb.set_trace()
        return tour, -calc_op_total(td['prize'].tolist(), tour)


# def calc_op_total(prize, tour):
#     # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
#     assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
#     assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
#     return np.array(prize)[np.array(tour) - 1].sum()

# def calc_op_length(depot, loc, tour):
#     assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
#     loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
#     sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
#     return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


# def write_compass_par(filename, parameters):
#     default_parameters = {  # Use none to include as flag instead of kv
#         "SPECIAL": None,
#         "MAX_TRIALS": 10000,
#         "RUNS": 10,
#         "TRACE_LEVEL": 1,
#         "SEED": 0
#     }
#     with open(filename, 'w') as f:
#         for k, v in {**default_parameters, **parameters}.items():
#             if v is None:
#                 f.write("{}\n".format(k))
#             else:
#                 f.write("{} = {}\n".format(k, v))


# def write_oplib(filename, depot, loc, prize, max_length, name="problem"):

#     with open(filename, 'w') as f:
#         f.write("\n".join([
#             "{} : {}".format(k, v)
#             for k, v in (
#                 ("NAME", name),
#                 ("TYPE", "OP"),
#                 ("DIMENSION", len(loc) + 1),
#                 ("COST_LIMIT", int(max_length * SCALING_FACTOR + 0.5)),
#                 ("EDGE_WEIGHT_TYPE", "EUC_2D"),
#             )
#         ]))
#         f.write("\n")
#         f.write("NODE_COORD_SECTION\n")
#         f.write("\n".join([
#             "{}\t{}\t{}".format(i + 1, int(x * SCALING_FACTOR + 0.5), int(y * SCALING_FACTOR + 0.5))  # oplib does not take floats
#             #"{}\t{}\t{}".format(i + 1, x, y)
#             for i, (x, y) in enumerate([depot] + loc)
#         ]))
#         f.write("\n")
#         f.write("NODE_SCORE_SECTION\n")
#         f.write("\n".join([
#             "{}\t{}".format(i + 1, d)
#             for i, d in enumerate([0] + prize)
#         ]))
#         f.write("\n")
#         f.write("DEPOT_SECTION\n")
#         f.write("1\n")
#         f.write("-1\n")
#         f.write("EOF\n")

                
# def read_oplib(filename, n):
#     with open(filename, 'r') as f:
#         tour = []
#         dimension = 0
#         started = False
#         for line in f:
#             if started:
#                 loc = int(line)
#                 if loc == -1:
#                     break
#                 tour.append(loc)
#             if line.startswith("DIMENSION"):
#                 dimension = int(line.split(" ")[-1])

#             if line.startswith("NODE_SEQUENCE_SECTION"):
#                 started = True
    
#     assert len(tour) > 0, "Unexpected length"
#     tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
#     assert tour[0] == 0  # Tour should start with depot
#     assert tour[-1] != 0  # Tour should not end with depot
#     return tour[1:].tolist()

def calc_op_total(prize, tour):
    # Subtract 1 since vals index start with 0 while tour indexing starts with 1 as depot is 0
    assert (np.array(tour) > 0).all(), "Depot cannot be in tour"
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    return np.array(prize)[np.array(tour) - 1].sum()


def calc_op_length(depot, loc, tour):
    assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def write_compass_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_oplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("NODE_SEQUENCE_SECTION"):
                started = True
    
    assert len(tour) > 0, "Unexpected length"
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_oplib(filename, depot, loc, prize, max_length, name="problem"):

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "OP"),
                ("DIMENSION", len(loc) + 1),
                ("COST_LIMIT", int(max_length * 10000000 + 0.5)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # oplib does not take floats
            #"{}\t{}\t{}".format(i + 1, x, y)
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("NODE_SCORE_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, d)
            for i, d in enumerate([0] + prize)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")
