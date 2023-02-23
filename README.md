<div align="center">

# NCOBench

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>


</div>

## Description

Code repository for NCOBench. Based on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) best practices.

> Note: this is an early work in progress and subject to change!

## How to run

Install dependencies

```bash
# Clone project
git clone https://github.com/kaist-silab/ncobench && cd ncobench


# Automatically install dependencies with light the torch
pip install light-the-torch && python3 -m light_the_torch install --upgrade -r requirements.txt
```
The above script will [automatically install](https://github.com/pmeier/light-the-torch) PyTorch with the right GPU version for your system. Alternatively, you can use `pip install -r requirements.txt` 


Train model with default configuration
```bash
python ncobench/train.py
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# Change experiment
python ncobench/train.py experiment=tsp

# Create a sweep over hyperparameters (-m for multirun)
python ncobench/train.py -m experiment=tsp data.cfg.order=5,10,15,20,30,45,60
```