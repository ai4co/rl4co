import argparse
import itertools
import pickle

from collections import defaultdict
from pathlib import Path

import torch

from tqdm.auto import tqdm

from rl4co.tasks.eval import evaluate_policy
from rl4co.utils.lightning import load_model_from_checkpoint

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir", type=str, default=None)
    argparser.add_argument("--checkpoint", type=str, default="last.ckpt")
    argparser.add_argument("--config", type=str, default="config.yaml")
    argparser.add_argument("--out_dir", type=str, default="results/eval_methods")
    argparser.add_argument("--gpus", type=int, default=1)
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument(
        "--start_batch_size",
        type=int,
        default=4096,
        help="Batch size for scaling evaluation",
    )
    argparser.add_argument("--data_path", type=str, default=None)

    args = argparser.parse_args()

    cfg_path = Path(args.dir) / args.config
    checkpoint_path = Path(args.dir) / args.checkpoint

    print(
        "Loading model from config: {} and checkpoint: {}".format(
            cfg_path, checkpoint_path
        )
    )
    lit_module = load_model_from_checkpoint(cfg_path, checkpoint_path)

    # Set gpu number
    if args.gpus > 0:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        # print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
        device = torch.device("cuda:{}".format(str(args.gpu_id)))
        print("Device:", device)
    else:
        device = torch.device("cpu")

    if args.data_path is not None:
        lit_module.env.test_file = args.data_path
        lit_module.setup("test")

    # Setup
    env = lit_module.model.env
    policy = lit_module.model.policy.to(device)
    policy.eval()
    dataset = lit_module.test_dataset

    # Insert here experiments to run. Here we run different evaluation methods on the same
    # dataset with different sample sizes to campare the Pareto efficient ones

    # Evaluations
    num_augment_sizes = [1280]
    num_augment_after_multistart = [16]
    softmax_temps = [1.0]

    # Experiments (remove dihedral from main table)
    experiments = {
        "greedy": {},
        "augment": {"num_augment": num_augment_sizes},
        # "augment_dihedral_8": {"num_augment": 8, "force_dihedral": True},
        "sampling": {"samples": num_augment_sizes, "softmax_temp": softmax_temps},
        "greedy_multistart": {"num_starts": env.num_loc},
        # "greedy_multistart_augment_dihedral_8": {"num_starts": env.num_loc, "num_augment": 8, "force_dihedral": True},
        "greedy_multistart_augment": {
            "num_starts": env.num_loc,
            "num_augment": num_augment_after_multistart,
        },
    }

    # Evaluate all experiments
    results = defaultdict(list)

    for exp_name, exp_kwargs in tqdm(experiments.items(), desc="Total progress"):
        kwargs = {}

        # Make kwargs into list
        for k, v in exp_kwargs.items():
            if isinstance(v, list):
                kwargs[k] = v
            else:
                kwargs[k] = [v]

        # Make all combinations
        kwargs = list(itertools.product(*kwargs.values()))

        # set keys
        kwargs = [dict(zip(exp_kwargs.keys(), k)) for k in kwargs]

        for kws_single_exp in kwargs:
            tqdm.write("=====================================")
            tqdm.write(
                "Running experiment: {} with kwargs: {}".format(exp_name, kws_single_exp)
            )

            retvals = evaluate_policy(
                env,
                policy,
                dataset,
                method=exp_name,
                start_batch_size=args.start_batch_size,
                **kws_single_exp,
            )

            # Add to retvals the exp_name and exp_kwargs
            retvals["exp_name"] = exp_name
            retvals["exp_kwargs"] = kws_single_exp

            # Add to results
            results[exp_name].append(retvals)

    # Finish
    out_dir = Path(args.out_dir)

    # Take the names of the two parent of the checkpoint path
    # names of parent. e.g. /home/botu/Dev/file.npz -> botu/Dev
    parent1 = checkpoint_path.parent.parent.name
    parent2 = checkpoint_path.parent.name
    out_dir = out_dir / parent1 / parent2

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save results with pickle
    fpath = out_dir / "results.pkl"
    print("Saving results to: {}".format(fpath))
    pickle.dump(results, open(fpath, "wb"))
