# Training with Hydra Configurations

You may find Hydra configurations under [configs/](https://github.com/ai4co/rl4co/tree/main/configs) divided into categories (model, env, train, experiment, etc.).

In practice, we usually want to modify configurations under the `experiment` folder, of which we report an example below [here](#experiment-configuration-example).

## Usage

Train model with default configuration (AM on TSP environment):
```bash
python run.py
```


### Change experiment

Train model with chosen experiment configuration from [configs/experiment/](https://github.com/ai4co/rl4co/tree/main/configs/experiment)
```bash
python run.py experiment=routing/am env=tsp env.generator_params.num_loc=50 model.optimizer_kwargs.lr=2e-4
```
Here you may change the environment, e.g. with `env=cvrp` by command line or by modifying the corresponding experiment e.g. [configs/experiment/routing/am.yaml](https://github.com/ai4co/rl4co/tree/main/configs/experiment/routing/am.yaml).


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

## Experiment Configuration Example

We report here a configuration for running the Attention Model (AM) on a TSP environment with 50 locations that can be placed under `configs/experiment`:

```yaml linenums="1"
# @package _global_

defaults:
  - override /model: am.yaml
  - override /env: tsp.yaml
  - override /callbacks: default.yaml
  - override /trainer: default.yaml
  - override /logger: wandb.yaml

env:
  generator_params:
    loc_distribution: "uniform"
    num_loc: 50

model:
  policy:
    _target_: "rl4co.models.zoo.am.AttentionModelPolicy"
    embed_dim: 128
    num_heads: 8
    num_encoder_layers: 3
  batch_size: 512
  val_batch_size: 1024
  test_batch_size: 1024
  train_data_size: 1_280_000
  val_data_size: 10_000
  test_data_size: 10_000
  optimizer_kwargs:
    lr: 1e-4
    weight_decay: 1e-6
  lr_scheduler:
    "MultiStepLR"
  lr_scheduler_kwargs:
    milestones: [80, 95]
    gamma: 0.1

trainer:
  max_epochs: 100

logger:
  wandb:
    project: "rl4co"
    tags: ["am", "${env.name}"]
    group: ${env.name}${env.generator_params.num_loc}
    name: am-${env.name}${env.generator_params.num_loc}
```

What does this configuration do? Let's break it down!

```yaml linenums="3"
defaults:
  - override /model: am.yaml
  - override /env: tsp.yaml
  - override /callbacks: default.yaml
  - override /trainer: default.yaml
  - override /logger: wandb.yaml
```

This section sets the default configuration for the model, environment, callbacks, trainer, and logger. This means that if a key is not specified in the experiment configuration, the default value will be used. Note that these are set in the root [configs/](https://github.com/ai4co/rl4co/tree/main/configs) folder, and are useful for better organization and reusability.

```yaml linenums="11"
env: 
  generator_params:
    loc_distribution: "uniform"
    num_loc: 50
```

This section specifies the environment configuration. In this case, we are using the TSP environment with 50 locations generated uniformly.

```yaml linenums="16"
model:
  policy:
    _target_: "rl4co.models.zoo.am.AttentionModelPolicy"
    embed_dim: 128
    num_heads: 8
    num_encoder_layers: 3
  batch_size: 512
  val_batch_size: 1024
  test_batch_size: 1024
  train_data_size: 1_280_000
  val_data_size: 10_000
  test_data_size: 10_000
  optimizer_kwargs:
    lr: 1e-4
    weight_decay: 1e-6
  lr_scheduler:
    "MultiStepLR"
  lr_scheduler_kwargs:
    milestones: [80, 95]
    gamma: 0.1
```

This section specifies the RL model (i.e., Lightning module) configuration. While this usually includes the policy architecture already (hence the name "model"), we can override it by specifying a `_target_` key and additional parameters to initialize the policy. Finally, we specify the batch sizes, data sizes, optimizer parameters, and learning rate scheduler.

```yaml linenums="37"
trainer:
  max_epochs: 100
```

This section specifies the trainer configuration. Here, we are training the model for 100 epochs.

```yaml linenums="40"
logger:
  wandb:
    project: "rl4co"
    tags: ["am", "${env.name}"]
    group: ${env.name}${env.generator_params.num_loc}
    name: am-${env.name}${env.generator_params.num_loc}
```

Finally, this section specifies the logger configuration. In this case, we are using Weights & Biases (WandB) to log the results of the experiment. We specify the project name, tags, group, and name of the experiment.

That's it! 🎉 


!!! tip
    For more advanced content and detailed descriptions, you may also check out [this notebook](../../../examples/advanced/1-hydra-config.ipynb)!


Now, you are ready to start training. If you save the above under `configs/experiment/mynewexperiment.yaml`, you can run it from the root of your RL4CO-based project with:
```bash
python run.py experiment=mynewexperiment
```