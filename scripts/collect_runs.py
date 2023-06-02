import sys
import os
import glob


if __name__ == "__main__":


    env_names = ['tsp20', 'tsp50', 'cvrp20', 'cvrp50']
    models = ['am', 'pomo', 'symnco']

    # Get all exps folders
    exps = [model+'-'+env_name for model in models for env_name in env_names]
    exps.extend(['am-'+env_name+'-sm'  for env_name in env_names])
    exps.extend(['am-'+env_name+'-sm-xl'  for env_name in env_names])

    # Checkpoint sizes: get last epoch and epoch such that sample sizes are normalized
    {   'am-*': [49, 99],
        'pomo-*': [15, 99],
        'symnco-*': [9, 99],
        'am-*-sm*': [49, 499],
    }

    # get save folder as current directory
    save_folder = os.path.join(os.getcwd(), 'checkpoints')

    # Go into logs/train/runs folder and collect all the runs
    for env_name in env_names:
        # search if there is a folder with env_name
        path = os.path.join('logs', 'train', env_name, 'runs')
        if os.path.exists(path):
            # go inside the folder and list all the runs
            os.chdir(path)
            # get all the runs
            runs = glob.glob('*')


