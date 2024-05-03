# Transductive Models


Transductive models are learning algorithms that optimize on a specific instance. They improve solutions by updating policy parameters $\theta$_, which means that we are running optimization (backprop) **at test time**.  Transductive learning can be performed with different policies: for example EAS updates (a part of) AR policies parameters to obtain better solutions, but I guess there are ways (or papers out there I don't know of) that optimize at test time.


```{eval-rst}
.. tip::
   You may refer to the definition of `inductive vs transductive RL <https://en.wikipedia.org/wiki/Transduction_(machine_learning)>`_. In inductive RL, we train to generalize to new instances. In transductive RL we train (or finetune) to solve only specific ones.
```


## Base Transductive Model

```{eval-rst}
.. automodule:: rl4co.models.common.transductive.base
   :members:
   :undoc-members:
```