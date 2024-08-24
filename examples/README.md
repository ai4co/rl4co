# 🧩 Examples and Tutorials

This is a collection of examples and tutorials for using the RL4CO library.

The root directory is made of quickstarts and contains the following:


## ⚡️ Quickstarts

This is the root directory of the examples. The following quickstarts are available:
- [`1-quickstart.ipynb`](1-quickstart.ipynb): here we train a model on a simple environment - it takes less than 2 minutes!
- [`2-full-training.ipynb`](2-full-training.ipynb): similar to the previous notebooks but with a more interesting environment, with checkpointing, logging, and callbacks.

    - [`2b-train-simple.py`](2b-train-simple.py): here we show a simple script that can be called with `python 2b-train-simple.py`. This is simplified and _does not use Hydra_ - for those who prefer a simpler setup. Note that we also made a Hydra tutorial [here](advanced/1-hydra-config.ipynb).
- [`3-creating-new-env-model.ipynb`](3-creating-new-env-model.ipynb): here we show how to extend RL4CO to solve new problems and create new models from zero to hero!


## 📁 Folders Index

### Modeling
Under the [`modeling/`](modeling) directory, here are some additional examples for modeling and inference.

### Datasets
Under the [`datasets/`](datasets) directory, here are some additional examples for using your custom data to train/evaluate your models

### Advanced
Under the [`advanced/`](advanced) directory, here are some additional examples for advanced topics.

### Other
Under the [`other/`](other) directory, here are some additional examples for other topics.
