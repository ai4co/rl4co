# Installation

<a href="https://colab.research.google.com/github/kaist-silab/rl4co/blob/main/notebooks/1-quickstart.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

RL4CO is now available for installation on `pip`!
```bash
pip install rl4co
```

## Local install and development
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

To get started, we recommend checking out our [quickstart notebook](https://github.com/kaist-silab/rl4co/blob/main/notebooks/1-quickstart.ipynb) or the [minimalistic example](#minimalistic-example).