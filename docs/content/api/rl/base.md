# RL4COLitModule

The `RL4COLitModule` is a wrapper around PyTorch Lightning's `LightningModule` that provides additional functionality for RL algorithms. It is the parent class for all RL algorithms in the library.

::: models.rl.common.base.RL4COLitModule


## Transductive Learning

Transductive models are learning algorithms that optimize on a specific instance. They improve solutions by updating policy parameters $\theta$, which means that we are running optimization (backprop) **at test time**.  Transductive learning can be performed with different policies: for example EAS updates (a part of) AR policies parameters to obtain better solutions, but I guess there are ways (or papers out there I don't know of) that optimize at test time.


!!! tip
    You may refer to the definition of [inductive vs transductive RL](https://en.wikipedia.org/wiki/Transduction_(machine_learning)) . In inductive RL, we train to generalize to new instances. In transductive RL we train (or finetune) to solve only specific ones.


:::models.common.transductive.base
    options:
      show_root_heading: false