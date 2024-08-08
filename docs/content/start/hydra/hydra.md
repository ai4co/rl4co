# Training with Hydra Configuration

You may find Hydra configurations under [configs/](configs/) divided into categories (model, env, train, experiment, etc.).

## Usage

Train model with default configuration (AM on TSP environment):
```bash
python run.py
```

!!! tip
    You may check out [this notebook](examples/advanced/1-hydra-config.ipynb) to get started with Hydra!


### Change experiment

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```bash
python run.py experiment=routing/am env=tsp env.generator_params.num_loc=50 model.optimizer_kwargs.lr=2e-4
```
Here you may change the environment, e.g. with `env=cvrp` by command line or by modifying the corresponding experiment e.g. [configs/experiment/routing/am.yaml](configs/experiment/routing/am.yaml).
</details>


### Disable logging

```bash
python run.py experiment=test/am logger=none '~callbacks.learning_rate_monitor'
```
Note that `~` is used to disable a callback that would need a logger.


### Create a sweep over hyperparameters

We can use -m for multirun:

```bash
python run.py -m experiment=routing/am  model.optimizer_kwargs.lr=1e-3,1e-4,1e-5
```
```
</details>

