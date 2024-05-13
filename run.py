from rl4co.tasks.train import train

# Call the train function directly from inside the package
# You can also pass additional Hydra arguments, like:
# `python run.py experiment=routing/am env=cvrp env.num_loc=50`
# Alternatively, you may run without Hydra (see examples/1.quickstart.ipynb)
if __name__ == "__main__":
    train()
