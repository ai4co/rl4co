import argparse
import glob
import os
import re

# Checkpoint sizes: get last epoch and epoch such that sample sizes are normalized
checkpoint_epochs = {
    "am-*": ["049", "099"],
    "am-critic*": ["049", "099"],  # NOTE: we will not use the 049
    "pomo-*": ["015", "099"],
    "symnco-*": ["009", "099"],
    "am-*-sm*": ["049", "499"],
    "am-sm*": ["049", "499"],  # backup for am-*-sm
    "ptrnet-*": ["049", "099"],
}


def get_checkpoint_epochs(key):
    """Get checkpoint epochs for a given key with longest match"""
    matching_patterns = []

    for pattern in checkpoint_epochs:
        # regex match
        if re.match(pattern.replace("*", ".*"), key):
            matching_patterns.append(pattern)

    if len(matching_patterns) > 0:
        longest_match = max(matching_patterns, key=len)
        return checkpoint_epochs[longest_match]
    else:
        assert False, "No checkpoint epochs found for key: {}".format(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect and copy checkpoint files for specific experiments"
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="saved_checkpoints",
        help="Path to the folder where checkpoint files will be saved",
    )
    parser.add_argument(
        "--env_names",
        nargs="+",
        default=["tsp20", "tsp50", "cvrp20", "cvrp50"],
        help="List of environment names",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["am", "pomo", "symnco", "ptrnet"],
        help="List of models",
    )
    parser.add_argument(
        "--zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to zip the folder. Use --no-zip False to disable",
    )
    args = parser.parse_args()

    # Params
    save_folder = args.save_folder
    env_names = args.env_names
    models = args.models

    # Get all exps folders
    exps = [model + "-" + env_name for model in models for env_name in env_names]
    exps.extend(["am-" + env_name + "-sm" for env_name in env_names])
    exps.extend(["am-" + env_name + "-sm-xl" for env_name in env_names])
    exps.extend(["am-sm-" + env_name for env_name in env_names])
    exps.extend(["am-critic-" + env_name for env_name in env_names])

    # Go into logs/train/runs folder and collect all the runs
    for env_name in env_names:
        # search if there is a folder with env_name
        path = os.path.join("logs", "train", "runs", env_name)
        if os.path.exists(path):
            # iterate over exps
            for exp in exps:
                # Get exp number of epochs
                epoch_sizes = get_checkpoint_epochs(exp)
                assert epoch_sizes is not None, "epoch_sizes is None"

                # look recursively inside path/exp for all folders called checkpoint
                checkpoint_folders = glob.glob(
                    os.path.join(path, exp, "**", "checkpoints"), recursive=True
                )

                # if not found, continue
                if len(checkpoint_folders) == 0:
                    continue

                # for each checkpoint folder, check all the files inside. if files contain both
                # epoch_sizes[0] and epoch_sizes[1], then add the folder to the list

                experiment_files = []
                for checkpoint_folder in checkpoint_folders:
                    # get all files inside checkpoint folder
                    files = glob.glob(os.path.join(checkpoint_folder, "*"))

                    # get all files that contain epoch_sizes[0] and epoch_sizes[1]
                    files = [
                        file
                        for file in files
                        if epoch_sizes[0] in file or epoch_sizes[1] in file
                    ]
                    # if files is not empty, add the folder to the list

                    if len(files) == 2:
                        print(
                            "Found checkpoint for experiment: {} in folder: {}".format(
                                exp, checkpoint_folder
                            )
                        )

                    experiment_files.extend(files)

                # check if experiment_files is empty
                if len(experiment_files) == 0:
                    continue

                # if there are files with the same name, get the date in the filepath
                # and keep the most recent one

                # get the date in the filepath
                dates = [file.split("/")[-3] for file in experiment_files]

                # filter out duplicates
                dates = list(set(dates))

                # get the most recent date
                most_recent_date = max(dates)

                if len(dates) > 1:
                    print("Found multiple checkpoints for experiment: {}".format(exp))
                    print("Dates: {} ; you may want to check manually.".format(dates))
                    print(
                        "Will default to the most recent one: {}".format(most_recent_date)
                    )

                # filter out files that do not contain the most recent date
                experiment_files = [
                    file for file in experiment_files if most_recent_date in file
                ]

                # subfolder
                subfolder = os.path.join(save_folder, env_name, exp)

                # create folder if it does not exist elseskip
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder, exist_ok=True)

                # Load config.yaml. This file is under ../wandb/latest-run
                # and contains the hyperparameters of the experiment
                config_file = os.path.join(
                    os.path.dirname(os.path.dirname(experiment_files[0])),
                    "wandb/latest-run/files/config.yaml",
                )
                experiment_files.append(config_file)

                # copy files to save_folder
                for file in experiment_files:
                    os.system("cp {} {}".format(file, subfolder))
                    print("Copied file: {} to folder: {}".format(file, subfolder))

    # zip the folder
    if args.zip:
        print("Zipping folder: {}".format(save_folder))
        os.system("zip -r {}.zip {}".format(save_folder, save_folder))
