## Definition

Given a CO problem instance $\mathbf{x}$, we formulate the solution-generating procedure as a Markov Decision Process (MDP) characterized by a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$ as follows:

- **State** $\mathcal{S}$ is the space of states that represent the given problem $\mathbf{x}$ and the current partial solution being updated in the MDP.
- **Action** $\mathcal{A}$ is the action space, which includes all feasible actions $a_t$ that can be taken at each step $t$.
- **State Transition** $\mathcal{T}$ is the deterministic state transition function $s_{t+1} = \mathcal{T}(s_t, a_t)$ that updates a state $s_t$ to the next state $s_{t+1}$.
- **Reward** $\mathcal{R}$ is the reward function $\mathcal{R}(s_t, a_t)$ representing the immediate reward received after taking action $a_t$ in state $s_t$.
- **Discount Factor** $\gamma \in [0, 1]$ determines the importance of future rewards. Often, $\gamma = 1$ is used for CO problems, i.e., no discounting.

Since the state transition is deterministic, we represent the solution for a problem $\mathbf{x}$ as a sequence of $T$ actions $\mathbf{a} = (a_1, \ldots, a_T)$. Then the total return $\sum_{t=1}^T \mathcal{R}(s_t, a_t)$ translates to the negative cost function of the CO problem.


In the following, we define the above MDP for the main CO problem types we consider in our library.

## Routing Problems

Routing problems are perhaps the most known class of CO problems. They are problems of great practical importance, not only for logistics, where they are more commonly framed, but also for industry, engineering, science, and medicine. The typical objective of routing problems is to minimize the total length of the paths needed to visit some (or all) the nodes in a graph $G = (V, E)$., and $i, j \in V$ are nodes in the graph.

### MDP

For routing problems, in RL4CO we consider two types of MDPs: Construction MDP and Improvement MDP.

#### Construction MDP

The Construction MDP describes a process of iteratively building a solution from scratch:

- **State** $s_t \in \mathcal{S}$: Reflects (1) node-level information for each customer node (e.g., coordinates, demand), (2) global-level information about the route construction (e.g., remaining vehicle capacity), and (3) the current partial solution $\{a_1, \ldots, a_{t-1}\}$ where $a_i$ is the previously selected node (action) at time $i$. The initial state at $t = 0$ has an empty partial solution.

- **Action** $a_t \in \mathcal{A}$: Choosing a valid node from set $V$. The action space is state-dependent, with infeasible actions masked to ensure all constraints are satisfied.

- **Transition** $\mathcal{T}$: Deterministic transition that adds the selected action $a_t$ to the partial solution, updating it from $\{a_1, \ldots, a_{t-1}\}$ to $\{a_1, \ldots, a_{t-1}, a_t\}$, and updates the node-level and global-level information accordingly.

- **Reward** $\mathcal{R}$: Typically set to the negative value of the increase in tour length, ensuring that maximizing cumulative rewards is equivalent to minimizing the tour length objective.

- **Policy** $\pi$: Usually parameterized by a deep neural network, it decides on an action $a_t$ given the input state $s_t$. The policy is typically stochastic, learning an action distribution for selecting each node.

#### Improvement MDP

The Improvement MDP describes a search process similar to neighborhood search, starting from a sub-optimal solution $\bm{a}^{0}=(a_{0}^{0},\ldots, a_{T-1}^{0})$ and finding another one potentially with higher quality:

- **State** $s_t \in \mathcal{S}$: Reflects (1) node-level information for each customer node, (2) global-level information about the search (e.g., historical visited solutions and their costs), and (3) the current solution $\bm{a^t}$. The initial state $s_0$ contains a randomly generated feasible solution $\bm{a^0}$.

- **Action** $a_t \in \mathcal{A}$: A specific operation that changes the current solution $\bm{a^t}$ into a new one $\bm{a^{t+1}}$. For example, specifying two nodes $(i, j)$ in $V$ to perform a pairwise local search operation.

- **Transition** $\mathcal{T}$: Usually deterministic, accepting the proposed solution $\bm{a^{t+1}}$ as the solution for the next state and updating node-level and global-level information accordingly.

- **Reward** $\mathcal{R}$: Typically set to the immediate reduced objective value of the current best-so-far solution after taking the local search action.

- **Policy** $\pi$: Usually stochastic and parameterized by a deep model. The time horizon can be user-specified based on the available time budget, often requiring a discount factor $\gamma < 1$.

The best solution found throughout the improvement horizon is recognized as the final solution to the routing problem.



### Documentation
Click [here](../api/envs/routing.md) for API documentation on routing problems.



## Scheduling Problems

Scheduling problems are a fundamental class of problems in operations research and industrial engineering, where the objective is to optimize the allocation of resources over time. These problems are critical in various industries, such as manufacturing, computer science, and project management. 



### MDP

Here we show a general constructive MDP formulation based on the Job Shop Scheduling Problem (JSSP), a well-known scheduling problem, which can be adapted to other scheduling problems.

- **State** $s_t \in \mathcal{S}$: 
  The state is represented by a disjunctive graph, where:
  - Operations are nodes
  - Processing orders between operations are shown by directed arcs
  - This graph encapsulates both the problem instance and the current partial schedule

- **Action** $a_t \in \mathcal{A}$: 
  An action involves selecting a feasible operation to assign to its designated machine, a process often referred to as dispatching. The action space consists of all operations that can be feasibly scheduled at the current state.

- **Transition** $\mathcal{T}$: 
  The transition function deterministically updates the disjunctive graph based on the dispatched operation. This includes:
  - Modifying the graph's topology (e.g., adding new connections between operations)
  - Updating operation attributes (e.g., start times)

- **Reward** $\mathcal{R}$: 
  The reward function is designed to align with the optimization objective. For instance, if minimizing makespan is the goal, the reward could be the negative change in makespan resulting from the latest action.

- **Policy** $\pi$: 
  The policy, typically stochastic, takes the current disjunctive graph as input and outputs a probability distribution over feasible dispatching actions. This process continues until a complete schedule is constructed.



### Documentation
Click [here](../api/envs/scheduling.md) for API documentation on scheduling problems.


## Electronic Design Automation

Electronic Design Automation (EDA) is a sophisticated process that involves the use of software tools to design, simulate, and analyze electronic systems, particularly integrated circuits (ICs) and printed circuit boards (PCBs). EDA encompasses a wide range of tasks, from schematic capture and layout design to verification and testing. Optimization is a critical aspect of EDA, where the goal is to achieve the best possible performance, power efficiency, and cost within the constraints of the design.

### MDP

EDA encompasses many problem types; here we'll focus on placement problems, which are fundamental in the physical design of integrated circuits and printed circuit boards. We'll use the Decap Placement Problem (DPP) as an example to illustrate a typical MDP formulation for EDA placement problems.


- **State** $s_t \in \mathcal{S}$: 
  The state typically represents the current configuration of the design space, which may include:
  - Locations of fixed elements (e.g., ports, keepout regions)
  - Current placements of movable elements
  - Remaining resources or components to be placed

- **Action** $a_t \in \mathcal{A}$: 
  An action usually involves placing a component at a valid location within the design space. The action space consists of all feasible placement locations, considering design rules and constraints.

- **Transition** $\mathcal{T}$: 
  The transition function updates the design state based on the placement action, which may include:
  - Updating the placement map
  - Adjusting available resources or remaining components
  - Recalculating relevant metrics (e.g., wire length, power distribution)

- **Reward** $\mathcal{R}$: 
  The reward is typically based on the improvement in the design objective resulting from the latest placement action. This could involve metrics such as area efficiency, signal integrity, or power consumption.

- **Policy** $\pi$: 
  The policy takes the current design state as input and outputs a probability distribution over possible placement actions.

Note that specific problems may introduce additional complexities or constraints.


### Documentation
Click [here](../api/envs/eda.md) for API documentation on EDA problems.


## Graph Problems

Many CO problems can be (re-)formulated on graphs. In typical CO problems on graphs, actions are defined on nodes/edges, while problem variables and constraints are incorporated in graph topology and node/edge attributes (e.g., weights). The graph-based formulation gives us concise and systematic representations of CO problems.

In graph problems, we typically work with a graph $G = (V, E)$, where $V$ is a set of vertices (or nodes) and $E$ is a set of edges connecting these vertices. The optimization task often involves selecting a subset of vertices, edges, or subgraphs to maximize or minimize a given objective function, subject to certain constraints.


### MDP

Graph problems can be effectively modeled using a Markov Decision Process (MDP) framework in a constructive fashion. Here, we outline the key components of the MDP formulation for graph problems:

- **State** $s_t \in \mathcal{S}$: 
  The state encapsulates the current configuration of the graph and the optimization progress. It typically includes:
  - The graph structure (vertices and edges)
  - Attributes associated with vertices or edges
  - The set of elements (vertices, edges, or subgraphs) selected so far
  - Problem-specific information, such as remaining selections or resources

- **Action** $a_t \in \mathcal{A}$: 
  An action usually involves selecting a graph element (e.g., a vertex, edge, or subgraph). The action space comprises all valid selections based on the problem constraints and the current state.

- **Transition** $\mathcal{T}$: 
  The transition function $\mathcal{T}(s_t, a_t) \rightarrow s_{t+1}$ updates the graph state based on the selected action. This typically involves:
  - Updating the set of selected elements
  - Modifying graph attributes affected by the selection
  - Updating problem-specific information (e.g., remaining selections or resources)

- **Reward** $\mathcal{R}$: 
  The reward function $\mathcal{R}(s_t, a_t)$ quantifies the quality of the action taken. It is typically based on the improvement in the optimization objective resulting from the latest selection. This could involve metrics such as coverage, distance, connectivity, or any other problem-specific criteria.

- **Policy** $\pi$: 
  The policy $\pi(a_t|s_t)$ is a probability distribution over possible actions given the current state. It guides the decision-making process, determining which graph elements to select at each step to optimize the objective.

Specific problems may introduce additional complexities or constraints, which can often be incorporated through careful design of the state space, action space, and reward function.


### Documentation
Click [here](../api/envs/graph.md) for API documentation on graph problems.



---


## Implementation Details

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
