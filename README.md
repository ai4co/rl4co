<div align="center">

# RL4CO


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/pytorch/rl"><img alt="base: TorchRL" src="https://img.shields.io/badge/base-TorchRL-red">
<a href="https://hydra.cc/"><img alt="config: Hydra" src="https://img.shields.io/badge/config-Hydra-89b8cd"></a> [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![Slack](https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack)](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ)
![license](https://img.shields.io/badge/license-Apache%202.0-green.svg?)<a href="https://colab.research.google.com/github/kaist-silab/rl4co/blob/main/notebooks/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>[![PyPI](https://img.shields.io/pypi/v/rl4co?logo=pypi)](https://pypi.org/project/rl4co)
[![Test](https://github.com/kaist-silab/rl4co/actions/workflows/tests.yml/badge.svg)](https://github.com/kaist-silab/rl4co/actions/workflows/tests.yml)
<!-- ![testing](https://github.com/kaist-silab/ncobench/actions/workflows/tests.yml/badge.svg) -->

[**Documentation**](https://rl4co.readthedocs.io/) |  [**Getting Started**](#getting-started) | [**Usage**](#usage) | [**Contributing**](#contributing) | [**Paper**](https://arxiv.org/abs/2306.17100) | [**Citation**](#cite-us)

</div>

---


An extensive Reinforcement Learning (RL) for Combinatorial Optimization (CO) benchmark. Our goal is to provide a unified framework for RL-based CO algorithms, and to facilitate reproducible research in this field, decoupling the science from the engineering.


RL4CO is built upon:
- [TorchRL](https://github.com/pytorch/rl): official PyTorch framework for RL algorithms and vectorized environments on GPUs
- [TensorDict](https://github.com/pytorch-labs/tensordict): a library to easily handle heterogeneous data such as states, actions and rewards
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning): a lightweight PyTorch wrapper for high-performance AI research
- [Hydra](https://github.com/facebookresearch/hydra): a framework for elegantly configuring complex applications

![image](https://github.com/kaist-silab/rl4co/assets/48984123/0db4efdd-1c93-4991-8f09-f3c6c1f35d60)


## Getting started
<a href="https://colab.research.google.com/github/kaist-silab/rl4co/blob/main/notebooks/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

RL4CO is now available for installation on `pip`!
```bash
pip install rl4co
```

### Local install and development
If you want to develop RL4CO or access the latest builds, we recommend you to install it locally with `pip` in editable mode:

```bash
git clone https://github.com/kaist-silab/rl4co && cd rl4co
pip install -e .
```
<details>
    <summary>[Optional] Automatically install PyTorch with correct CUDA version</summary>

These commands will [automatically install](https://github.com/pmeier/light-the-torch) PyTorch with the right GPU version for your system:

```bash
pip install light-the-torch
python3 -m light_the_torch install -r  --upgrade torch
```

> Note: `conda` is also a good candidate for hassle-free installation of PyTorch: check out the [PyTorch website](https://pytorch.org/get-started/locally/) for more details.

</details>



To get started, we recommend checking out our [quickstart notebook](notebooks/1-quickstart.ipynb) or the [minimalistic example](#minimalistic-example) below.

## Usage


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



### Minimalistic Example

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


### Testing

Run tests with `pytest` from the root directory:

```bash
pytest tests
```

## Contributing
[![Slack](https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack)](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ)

Have a suggestion, request, or found a bug? Feel free to [open an issue](https://github.com/kaist-silab/rl4co/issues) or [submit a pull request](https://github.com/kaist-silab/rl4co/pulls).
If you would like to contribute, please check out our contribution guidelines   [here](.github/CONTRIBUTING.md). We welcome and look forward to all contributions to RL4CO!

We are also on [Slack](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ) if you have any questions or would like to discuss RL4CO with us. We are open to collaborations and would love to hear from you 🚀


### Contributors
<a href="https://github.com/kaist-silab/rl4co/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kaist-silab/rl4co" />
</a>

## Cite us
If you find RL4CO valuable for your research or applied projects:

```bibtex
@article{berto2023rl4co,
    title = {{RL4CO}: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Minsu Kim and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Joungho Kim and Jinkyoo Park},
    journal={arXiv preprint arXiv:2306.17100},
    year={2023},
    url = {https://github.com/kaist-silab/rl4co}
}
```
