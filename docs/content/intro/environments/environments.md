## Definition

Given a CO problem instance $\mathbf{x}$, we formulate the solution-generating procedure as a Markov Decision Process (MDP) characterized by a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$ as follows:

- **State** $\mathcal{S}$ is the space of states that represent the given problem $\mathbf{x}$ and the current partial solution being updated in the MDP.
- **Action** $\mathcal{A}$ is the action space, which includes all feasible actions $a_t$ that can be taken at each step $t$.
- **State Transition** $\mathcal{T}$ is the deterministic state transition function $s_{t+1} = \mathcal{T}(s_t, a_t)$ that updates a state $s_t$ to the next state $s_{t+1}$.
- **Reward** $\mathcal{R}$ is the reward function $\mathcal{R}(s_t, a_t)$ representing the immediate reward received after taking action $a_t$ in state $s_t$.
- **Discount Factor** $\gamma \in [0, 1]$ determines the importance of future rewards.

Since the state transition is deterministic, we represent the solution for a problem $\mathbf{x}$ as a sequence of $T$ actions $\mathbf{a} = (a_1, \ldots, a_T)$. Then the total return $\sum_{t=1}^T \mathcal{R}(s_t, a_t)$ translates to the negative cost function of the CO problem.

## Implementation

Environments in our library fully specify the CO problems and their logic. They are based on the `RL4COEnvBase` class that extends from the [`EnvBase`](https://pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) in TorchRL.

Key features:
- A modular `generator` can be provided to the environment.
- The generator provides CO instances to the environment, and different generators can be used to generate different data distributions.
- Static instance data and dynamic variables, such as the current state $s_t$, current solution $\mathbf{a}^k$ for improvement environments, policy actions $a_t$, rewards, and additional information are passed in a *stateless* fashion in a [`TensorDict`](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html?highlight=tensordict#tensordict.TensorDict), that we call `td`, through the environment `reset` and `step` functions.

Our environment API contains several functions:
- `render`
- `check_solution_validity`
- `select_start_nodes` (i.e., for POMO-based optimization)
- Optional API such as `local_search` for solution improvement

It's worth noting that our library enhances the efficiency of environments when compared to vanilla TorchRL, by overriding and optimizing some methods in TorchRL [`EnvBase`](https://pytorch.org/rl/stable/reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase). For instance, our new `step` method brings a decrease of up to 50% in latency and halves the memory impact by avoiding saving duplicate components in the stateless [`TensorDict`](https://pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html?highlight=tensordict#tensordict.TensorDict).