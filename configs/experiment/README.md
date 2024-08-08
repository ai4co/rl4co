# Experiments


> [!NOTE] 
> To reproduce results in the latest version of the paper, please use the most recent version as described below. 

In the latest experiments version (as of the NeurIPS GLFrontiers WS paper) we used experiments in the [routing](routing/) folder, which  is more modular then before.

To change the experiment task, you can simply call the model and change the `env` to the target such as:
    
```python
python run.py experiment=routing/pomo env=op 
```

> [!TIP] 
> Stay tuned for the upcoming NeurIPS 2024 released, several new experiments are coming soon!


---


## Older experiments

Older version (around `v0.0.3`) experiments are under the [older experiments](archive/README.md). Note that there can be rough edges and you may need to install an older version of the library to run these experiments.