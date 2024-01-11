<div align="center">


<img src="https://github.com/ai4co/rl4co/assets/48984123/01a547b2-9722-4540-b0e1-9c12af094b15" style="width:40%">


</br></br>


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/pytorch/rl"><img alt="base: TorchRL" src="https://img.shields.io/badge/base-TorchRL-red">
<a href="https://hydra.cc/"><img alt="config: Hydra" src="https://img.shields.io/badge/config-Hydra-89b8cd"></a> [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Slack](https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack)](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ)
![license](https://img.shields.io/badge/license-Apache%202.0-green.svg?) <a href="https://colab.research.google.com/github/ai4co/rl4co/blob/main/notebooks/1-quickstart.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> [![PyPI](https://img.shields.io/pypi/v/rl4co?logo=pypi)](https://pypi.org/project/rl4co)
[![Test](https://github.com/ai4co/rl4co/actions/workflows/tests.yml/badge.svg)](https://github.com/ai4co/rl4co/actions/workflows/tests.yml)

[**Documentation**](https://rl4co.readthedocs.io/) |  [**Getting Started**](#getting-started) | [**Usage**](#usage) | [**Contributing**](#contributing) | [**Paper**](https://arxiv.org/abs/2306.17100) | [**Join Us**](#join-us)

</div>

---

RL4CO has been accepted as an oral presentation at the [NeurIPS 2023 GLFrontiers Workshop](https://glfrontiers.github.io/)! 🎉

---


An extensive Reinforcement Learning (RL) for Combinatorial Optimization (CO) benchmark. Our goal is to provide a unified framework for RL-based CO algorithms, and to facilitate reproducible research in this field, decoupling the science from the engineering.


RL4CO is built upon:
- [TorchRL](https://github.com/pytorch/rl): official PyTorch framework for RL algorithms and vectorized environments on GPUs
- [TensorDict](https://github.com/pytorch-labs/tensordict): a library to easily handle heterogeneous data such as states, actions and rewards
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning): a lightweight PyTorch wrapper for high-performance AI research
- [Hydra](https://github.com/facebookresearch/hydra): a framework for elegantly configuring complex applications

![RL4CO Overview](https://github.com/ai4co/rl4co/assets/34462374/4d9a670f-ab7c-4fc8-9135-82d17cb6d0ee)


We provide several utilities and modularization. For autoregressive policies, we modularize reusable components such as _environment embeddings_ that can easily be swapped to [solve new problems](https://github.com/ai4co/rl4co/blob/main/notebooks/tutorials/2-creating-new-env-model.ipynb)
![RL4CO Policy](https://github.com/ai4co/rl4co/assets/48984123/ca88f159-d0b3-459e-8fd9-89799be9d1b0)

## Getting started
<a href="https://colab.research.google.com/github/ai4co/rl4co/blob/main/notebooks/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

RL4CO is now available for installation on `pip`!
```bash
pip install rl4co
```

To get started, we recommend checking out our [quickstart notebook](notebooks/1-quickstart.ipynb) or the [minimalistic example](#minimalistic-example) below.

### Install from source
This command installs the bleeding edge `main` version, useful for staying up-to-date with the latest developments - for instance, if a bug has been fixed since the last official release but a new release hasn’t been rolled out yet:

```bash
pip install -U git+https://github.com/ai4co/rl4co.git
```

### Local install and development
If you want to develop RL4CO we recommend you to install it locally with `pip` in editable mode:

```bash
git clone https://github.com/ai4co/rl4co && cd rl4co
pip install -e .
```

We recommend using a virtual environment such as `conda` to install `rl4co` locally.



## Usage


Train model with default configuration (AM on TSP environment):
```bash
python run.py
```


<details>
    <summary>Change experiment</summary>

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/) (e.g. tsp/am, and environment with 42 cities)
```bash
python run.py experiment=routing/am env.num_loc=42
```
</details>


<details>
    <summary>Disable logging</summary>

```bash
python run.py experiment=routing/am logger=none '~callbacks.learning_rate_monitor'
```
Note that `~` is used to disable a callback that would need a logger.

</details>


<details>
    <summary>Create a sweep over hyperparameters (-m for multirun)</summary>

```bash
python run.py -m experiment=routing/am  train.optimizer.lr=1e-3,1e-4,1e-5
```
</details>



### Minimalistic Example

Here is a minimalistic example training the Attention Model with greedy rollout baseline on TSP in less than 30 lines of code:

```python
from rl4co.envs import TSPEnv
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer

# Environment, Model, and Lightning Module
env = TSPEnv(num_loc=20)
model = AttentionModel(env,
                       baseline="rollout",
                       train_data_size=100_000,
                       test_data_size=10_000,
                       optimizer_kwargs={'lr': 1e-4}
                       )

# Trainer
trainer = RL4COTrainer(max_epochs=3)

# Fit the model
trainer.fit(model)

# Test the model
trainer.test(model)
```

Other examples can be found on the [documentation](https://rl4co.readthedocs.io/en/latest/)!


### Testing

Run tests with `pytest` from the root directory:

```bash
pytest tests
```

### Known Bugs


#### Bugs installing PyTorch Geometric (PyG)

Installing `PyG` via `Conda` seems to update Torch itself. We have found that this update introduces some bugs with `torchrl`. At this moment, we recommend installing `PyG` with `Pip`:
```bash
pip install torch_geometric
```


## Contributing

Have a suggestion, request, or found a bug? Feel free to [open an issue](https://github.com/ai4co/rl4co/issues) or [submit a pull request](https://github.com/ai4co/rl4co/pulls).
If you would like to contribute, please check out our contribution guidelines   [here](.github/CONTRIBUTING.md). We welcome and look forward to all contributions to RL4CO!

We are also on [Slack](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ) if you have any questions or would like to discuss RL4CO with us. We are open to collaborations and would love to hear from you 🚀

### Contributors
<a href="https://github.com/ai4co/rl4co/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ai4co/rl4co" />
</a>

## Citation
If you find RL4CO valuable for your research or applied projects:

```bibtex
@inproceedings{berto2023rl4co,
    title={{RL}4{CO}: a Unified Reinforcement Learning for Combinatorial Optimization Library},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Minsu Kim and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Joungho Kim and Jinkyoo Park},
    booktitle={NeurIPS 2023 Workshop: New Frontiers in Graph Learning},
    year={2023},
    url={https://openreview.net/forum?id=YXSJxi8dOV},
    note={\url{https://github.com/ai4co/rl4co}}
}
```

## Join us
[![Slack](https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack)](https://join.slack.com/t/rl4co/shared_invite/zt-1ytz2c1v4-0IkQ8NQH4TRXIX8PrRmDhQ)

We invite you to join our AI4CO community, an open research group in Artificial Intelligence (AI) for Combinatorial Optimization (CO)!



<p align="center">
  <img width="30%" src="https://github.com/ai4co/rl4co/assets/48984123/2f1298ef-15e1-4a66-9741-78ee75938789">
</p>


