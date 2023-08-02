# Training with Hydra Configuration

You may find Hydra configurations under [configs/](configs/) divided into categories (model, env, train, experiment, etc.).

## Usage

Train model with default configuration (AM on TSP environment):
```bash
python run.py
```

### Change experiment

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/) (e.g. tsp/am, and environment with 42 cities)
```bash
python run.py experiment=tsp/am env.num_loc=42
```
</details>


### Disable logging

```bash
python run.py experiment=test/am logger=none '~callbacks.learning_rate_monitor'
```
Note that `~` is used to disable a callback that would need a logger.


### Create a sweep over hyperparameters

We can use -m for multirun:

```bash
python run.py -m experiment=tsp/am  train.optimizer.lr=1e-3,1e-4,1e-5
```
</details>

