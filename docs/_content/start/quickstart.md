# Quickstart to `rl4co`

Train model with default configuration (AM on TSP environment):
```bash
python run.py
```

<details>
    <summary>Change experiment</summary>

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/) (e.g. tsp/am, and environment with 42 cities)
```bash
python run.py experiment=tsp/am env.num_loc=42
```
</details>


<details>
    <summary>Disable logging</summary>

```bash
python run.py experiment=test/am logger=none '~callbacks.learning_rate_monitor'
```
Note that `~` is used to disable a callback that would need a logger.

</details>


<details>
    <summary>Create a sweep over hyperparameters (-m for multirun)</summary>

```bash
python run.py -m experiment=tsp/am  train.optimizer.lr=1e-3,1e-4,1e-5
```
</details>



## Minimalistic Example

Here is a minimalistic example training the Attention Model with greedy rollout baseline on TSP in less than 50 lines of code:

```python
from omegaconf import DictConfig
import lightning as L
from rl4co.envs import TSPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.tasks.rl4co import RL4COLitModule

config = DictConfig(
    {"data": {
            "train_size": 100000,
            "val_size": 10000,
            "batch_size": 512,
        },
    "optimizer": {"lr": 1e-4}}
)

# Environment, Model, and Lightning Module
env = TSPEnv(num_loc=20)
model = AttentionModel(env)
lit_module = RL4COLitModule(config, env, model)

# Trainer
trainer = L.Trainer(
    max_epochs=3, # only few epochs
    accelerator="gpu", # use GPU if available, else you can use others as "cpu"
    logger=None, # can replace with WandbLogger, TensorBoardLogger, etc.
    precision="16-mixed", # Lightning will handle faster training with mixed precision
    gradient_clip_val=1.0, # clip gradients to avoid exploding gradients
    reload_dataloaders_every_n_epochs=1, # necessary for sampling new data
)

# Fit the model
trainer.fit(lit_module)
```


## Testing

Run tests with `pytest` from the root directory:

```bash
pytest tests
```