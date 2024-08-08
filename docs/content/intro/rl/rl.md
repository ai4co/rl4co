# RL Algorithms


## Definitions

The RL objective is to learn a policy $\pi$ that maximizes the expected cumulative reward (or equivalently minimizes the cost) over the distribution of problem instances:

$$
\theta^{*} = \underset{\theta}{\text{argmax}} \, \mathbb{E}_{\mathbf{x} \sim P(\mathbf{x})} \left[ \mathbb{E}_{\pi(\mathbf{a}|\mathbf{x})} \left[ \sum_{t=0}^{T-1} \gamma^t \mathcal{R}(s_t, a_t) \right] \right],
$$

where $\theta$ is the set of parameters of $\pi$ and $P(\mathbf{x})$ is the distribution of problem instances.

This equation can be solved using algorithms such as variations of REINFORCE, Advantage Actor-Critic (A2C) methods, or Proximal Policy Optimization (PPO).

These algorithms are employed to train the policy network $\pi$, by transforming the maximization problem into a minimization problem involving a loss function, which is then optimized using gradient descent algorithms. For instance, the REINFORCE loss function gradient is given by:

$$
\nabla_{\theta} \mathcal{L}_a(\theta|\mathbf{x}) = \mathbb{E}_{\pi(\mathbf{a}|\mathbf{x})} \left[(R(\mathbf{a}, \mathbf{x}) - b(\mathbf{x})) \nabla_{\theta}\log \pi(\mathbf{a}|\mathbf{x})\right],
$$

where $b(\cdot)$ is a baseline function used to stabilize training and reduce gradient variance. 

We also distinguish between two types of RL (pre)training:

1. *Inductive RL*: The focus is on learning patterns from the training dataset to generalize to new instances, thus amortizing the inference procedure.
2. *Transductive RL* (or test-time optimization): Optimizes parameters during testing on target instances.

Typically, a policy $\pi$ is trained using inductive RL, followed by transductive RL for test-time optimization.

### Implementation

RL algorithms in our library define the process that takes the `Environment` with its problem instances and the `Policy` to optimize its parameters $\theta$. The parent class of algorithms is the `RL4COLitModule`, inheriting from PyTorch Lightning's [`pl.LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html). This allows for granular support of various methods including the `[train, val, test]_step`, automatic logging with several logging services such as Wandb via `log_metrics`, automatic optimizer configuration via `configure_optimizers` and several useful callbacks for RL methods such as `on_train_epoch_end`.

RL algorithms are additionally attached to an `RL4COTrainer`, a wrapper we made with additional optimizations around `pl.Trainer`. This module seamlessly supports features of modern training pipelines, including:

- Logging
- Checkpoint management
- Mixed-precision training
- Various hardware acceleration supports (e.g., CPU, GPU, TPU, and Apple Silicon)
- Multi-device hardware accelerator in distributed settings

For instance, using mixed-precision training significantly decreases training time without sacrificing much convergence and enables us to leverage recent routines, e.g., FlashAttention.