# RL4CO

<div align="center">
<p id="mainpage-title">Welcome to RL4CO</p>

An extensive Reinforcement Learning (RL) for Combinatorial Optimization (CO) benchmark. Our goal is to provide a unified framework for RL-based CO algorithms, and to facilitate reproducible research in this field, decoupling the science from the engineering.

<a href="https://pytorch.org/get-started/locally/"><img class="badge-tag" alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/pytorch/rl"><img alt="base: TorchRL" src="https://img.shields.io/badge/base-TorchRL-red">
<a href="https://hydra.cc/"><img alt="config: Hydra" src="https://img.shields.io/badge/config-Hydra-89b8cd"></a> [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<a href="https://github.com/kaist-silab/rl4co/blob/main/LICENSE">![license](https://img.shields.io/badge/license-Apache%202.0-green.svg?)</a>[![PyPI](https://img.shields.io/pypi/v/rl4co?logo=pypi)](https://pypi.org/project/rl4co)
[![Test](https://github.com/kaist-silab/rl4co/actions/workflows/tests.yml/badge.svg)](https://github.com/kaist-silab/rl4co/actions/workflows/tests.yml)

</div>


RL4CO is built upon:
- [TorchRL](https://github.com/pytorch/rl): official PyTorch framework for RL algorithms and vectorized environments on GPUs
- [TensorDict](https://github.com/pytorch-labs/tensordict): a library to easily handle heterogeneous data such as states, actions and rewards
- [PyTorch Lightning](https://github.com/Lightning-AI/lightning): a lightweight PyTorch wrapper for high-performance AI research
- [Hydra](https://github.com/facebookresearch/hydra): a framework for elegantly configuring complex applications

<img class="full-img" alt="image" src="https://github.com/kaist-silab/rl4co/assets/48984123/0db4efdd-1c93-4991-8f09-f3c6c1f35d60">



```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   _content/start/installation
   _content/start/quickstart
   _content/start/quickstart_notebook

.. toctree::
   :maxdepth: 2
   :caption: Algorithms:

   _content/api/algos/base
   _content/api/algos/reinforce
   _content/api/algos/ppo


.. toctree::
   :maxdepth: 2
   :caption: Environments:

   _content/api/envs/base
   _content/api/envs/eda
   _content/api/envs/routing
   _content/api/envs/scheduling

.. toctree::
   :maxdepth: 2
   :caption: Models:

   _content/api/models/base
   _content/api/models/zoo
   _content/api/models/nn
   _content/api/models/env_embeddings


.. toctree::
   :maxdepth: 2
   :caption: Additional API:

   _content/api/data

.. toctree::
   :maxdepth: 2
   :caption: General Information:

   _content/general/contribute
   _content/general/faq
```


## Contributors
<a href="https://github.com/kaist-silab/rl4co/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kaist-silab/rl4co" />
</a>