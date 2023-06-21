import argparse
import logging
import os
import sys

import numpy as np

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def generate_env_data(env_type, *args, **kwargs):
    """Generate data for a given environment type in the form of a dictionary"""
    try:
        # breakpoint()
        # remove all None values from args
        args = [arg for arg in args if arg is not None]

        return getattr(sys.modules[__name__], f"generate_{env_type}_data")(
            *args, **kwargs
        )
    except AttributeError:
        raise NotImplementedError(f"Environment type {env_type} not implemented")


def generate_tsp_data(dataset_size, tsp_size):
    return {
        "locs": np.random.uniform(size=(dataset_size, tsp_size, 2)).astype(np.float32)
    }


def generate_vrp_data(dataset_size, vrp_size, capacities=None):
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

    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    if capacities is not None:
        for k, v in capacities.items():
            if k in CAPACITIES:
                print(f"Replacing capacity for {k} with {v}")
                CAPACITIES[k] = v

    return {
        "depot": np.random.uniform(size=(dataset_size, 2)).astype(
            np.float32
        ),  # Depot location
        "locs": np.random.uniform(size=(dataset_size, vrp_size, 2)).astype(
            np.float32
        ),  # Node locations
        "demand": np.random.randint(1, 10, size=(dataset_size, vrp_size)).astype(
            np.float32
        ),  # Demand, uniform integer 1 ... 9
        "capacity": np.full(dataset_size, CAPACITIES[vrp_size]).astype(np.float32),
    }  # Capacity, same for whole dataset


def generate_op_data(dataset_size, op_size, prize_type="const"):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == "const":
        prize = np.ones((dataset_size, op_size))
    elif prize_type == "unif":
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.0
    else:  # Based on distance to depot
        assert prize_type == "dist"
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (
            1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)
        ) / 100.0

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}

    return {
        "depot": depot.astype(np.float32),
        "locs": loc.astype(np.float32),
        "prize": prize.astype(np.float32),
        "max_length": np.full(dataset_size, MAX_LENGTHS[op_size]).astype(np.float32),
    }


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {20: 2.0, 50: 3.0, 100: 4.0}
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = (
        np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)
    )

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = (
        np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2
    )

    return {
        "locs": loc.astype(np.float32),
        "depot": depot.astype(np.float32),
        "penalty": penalty.astype(np.float32),
        "deterministic_prize": deterministic_prize.astype(np.float32),
        "stochastic_prize": stochastic_prize.astype(np.float32),
    }


def generate_mdpp_data(
    dataset_size,
    size=10,
    num_probes_min=2,
    num_probes_max=5,
    num_keepout_min=1,
    num_keepout_max=50,
    lock_size=True,
):
    """Generate data for the nDPP problem.
    If `lock_size` is True, then the size if fixed and we skip the `size` argument if it is not 10.
    This is because the RL environment is based on a real-world PCB (parametrized with data)
    """
    if lock_size and size != 10:
        # log.info("Locking size to 10, skipping generate_mdpp_data with size {}".format(size))
        return None

    bs = dataset_size  # bs = batch_size to generate data in batch
    m = n = size
    if isinstance(bs, int):
        bs = [bs]

    locs = np.stack(np.meshgrid(np.arange(m), np.arange(n)), axis=-1).reshape(-1, 2)
    locs = locs / np.array([m, n], dtype=np.float32)
    locs = np.expand_dims(locs, axis=0)
    locs = np.repeat(locs, bs[0], axis=0)

    available = np.ones((bs[0], m * n), dtype=bool)

    probe = np.random.randint(0, high=m * n, size=(bs[0], 1))
    np.put_along_axis(available, probe, False, axis=1)

    num_probe = np.random.randint(num_probes_min, num_probes_max + 1, size=(bs[0], 1))
    probes = np.zeros((bs[0], m * n), dtype=bool)
    for i in range(bs[0]):
        p = np.random.choice(m * n, num_probe[i], replace=False)
        np.put_along_axis(available[i], p, False, axis=0)
        np.put_along_axis(probes[i], p, True, axis=0)

    num_keepout = np.random.randint(num_keepout_min, num_keepout_max + 1, size=(bs[0], 1))
    for i in range(bs[0]):
        k = np.random.choice(m * n, num_keepout[i], replace=False)
        np.put_along_axis(available[i], k, False, axis=0)

    return {
        "locs": locs.astype(np.float32),
        "probe": probes.astype(bool),
        "action_mask": available.astype(bool),
    }


def generate_dataset(
    filename=None,
    data_dir="data",
    name=None,
    problem="all",
    data_distribution="all",
    dataset_size=10000,
    graph_sizes=[20, 50, 100],
    overwrite=False,
    seed=1234,
    disable_warning=True,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict
    """
    assert filename is None or (
        len(problem) == 1 and len(graph_sizes) == 1
    ), "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        "tsp": [None],
        "vrp": [None],
        "pctsp": [None],
        "op": ["const", "unif", "dist"],
        "mdpp": [None],
    }
    if problem == "all":
        problems = distributions_per_problem
    else:
        problems = {
            problem: distributions_per_problem[problem]
            if data_distribution == "all"
            else [data_distribution]
        }
    # breakpoint()
    fname = filename
    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in graph_sizes:
                datadir = os.path.join(data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if filename is None:
                    fname = os.path.join(
                        datadir,
                        "{}{}{}_{}_seed{}.npz".format(
                            problem,
                            "_{}".format(distribution)
                            if distribution is not None
                            else "",
                            graph_size,
                            name,
                            seed,
                        ),
                    )
                else:
                    fname = check_extension(filename, extension=".npz")

                if not overwrite and os.path.isfile(
                    check_extension(fname, extension=".npz")
                ):
                    if not disable_warning:
                        log.info(
                            "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                                fname
                            )
                        )
                    continue

                # Set seed
                np.random.seed(seed)

                # Automatically generate dataset
                dataset = generate_env_data(
                    problem, dataset_size, graph_size, distribution
                )

                # A function can return None in case of an error or a skip
                if dataset is not None:
                    # Save to disk as dict
                    log.info("Saving {} dataset to {}".format(problem, fname))
                    np.savez(fname, **dataset)


def generate_default_datasets(data_dir):
    """Generate the default datasets used in the paper and save them to data_dir/problem"""
    generate_dataset(data_dir=data_dir, name="val", problem="all", seed=4321)
    generate_dataset(data_dir=data_dir, name="test", problem="all", seed=1234)
    generate_dataset(
        data_dir=data_dir,
        name="test",
        problem="mdpp",
        seed=1234,
        graph_sizes=[10],
        dataset_size=100,
    )  # EDA (mDPP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", help="Filename of the dataset to create (ignores datadir)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Create datasets in data_dir/problem (default 'data')",
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name to identify dataset"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
        " or 'all' to generate all",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="all",
        help="Distributions to generate for problem, default 'all'.",
    )
    parser.add_argument(
        "--dataset_size", type=int, default=10000, help="Size of the dataset"
    )
    parser.add_argument(
        "--graph_sizes",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Sizes of problem instances (default 20, 50, 100)",
    )
    parser.add_argument("-f", action="store_true", help="Set true to overwrite")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--disable_warning", action="store_true", help="Disable warning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    args.overwrite = args.f
    delattr(args, "f")
    generate_dataset(**vars(args))
