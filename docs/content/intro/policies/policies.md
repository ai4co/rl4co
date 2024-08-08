# Policies

The policies can be categorized into constructive policies, which generate a solution from scratch, and improvement policies, which refine an existing solution.

## Constructive policies

A policy $\pi$ is used to construct a solution from scratch for a given problem instance $\mathbf{x}$. It can be further categorized into autoregressive (AR) and non-autoregressive (NAR) policies.


### Autoregressive (AR) policies
An AR policy is composed of an encoder $f$ that maps the instance $\mathbf{x}$ into an embedding space $\mathbf{h}=f(\mathbf{x})$ and by a decoder $g$ that iteratively determines a sequence of actions $\mathbf{a}$ as follows:

$$
a_t \sim g(a_t | a_{t-1}, ... ,a_0, s_t, \mathbf{h}), \quad 
\pi(\mathbf{a}|\mathbf{x}) \triangleq \prod_{t=1}^{T-1} g(a_{t} | a_{t-1}, \ldots ,a_0, s_t, \mathbf{h}).
$$

### Non-autoregressive (NAR) policies
A NAR policy encodes a problem $\mathbf{x}$ into a heuristic $\mathcal{H} = f(\mathbf{x}) \in \mathbb{R}^{N}_{+}$, where $N$ is the number of possible assignments across all decision variables. Each number in $\mathcal{H}$ represents a (unnormalized) probability of a particular assignment. To obtain a solution $\mathbf{a}$ from $\mathcal{H}$, one can sample a sequence of assignments from $\mathcal{H}$ while dynamically masking infeasible assignments to meet problem-specific constraints. It can also guide a search process, e.g., Ant Colony Optimization, or be incorporated into hybrid frameworks. Here, the heuristic helps identify promising transitions and improve the efficiency of finding an optimal or near-optimal solution.


<div align="center">
<img src="https://github.com/ai4co/rl4co/assets/48984123/9e1f32f9-9884-49b9-b6cd-364861cc8fe7"  style="width: 100%; height: auto;">
</div>


## Improvement policies

A policy can be used for improving an initial solution $\mathbf{a}^{0}=(a_{0}^{0},\ldots, a_{T-1}^{0})$ into another one potentially with higher quality, which can be formulated as follows:

$$
\mathbf{a}^k \sim g(\mathbf{a}^{0}, \mathbf{h}), \quad\pi(\mathbf{a}^K|\mathbf{a}^0,\mathbf{x}) \triangleq \prod_{k=1}^{K-1} g(\mathbf{a}^k | \mathbf{a}^{k-1}, ... ,\mathbf{a}^0, \mathbf{h}),
$$

where $\mathbf{a}^{k}$ is the $k$-th updated solution and $K$ is the budget for number of improvements. This process allows continuous refinement for a long time to enhance the solution quality.




## Implementation

Policies in our library are subclasses of PyTorch's [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and contain the encoding-decoding logic and neural network parameters $\theta$. Different policies in the RL4CO "zoo" can inherit from metaclasses like `ConstructivePolicy` or `ImprovementPolicy`. We modularize components to process raw features into the embedding space via a parametrized function $\phi_\omega$, called *feature embeddings*.

1. *Node Embeddings $\phi_n$*: transform $m_n$ node features of instances $\mathbf{x}$ from the feature space to the embedding space $h$, i.e., $[B, N, m_n] \rightarrow [B, N, h]$.
2. *Edge Embeddings $\phi_e$*: transform $m_e$ edge features of instances $\mathbf{x}$ from the feature space to the embedding space $h$, i.e., $[B, E, m_e] \rightarrow [B, E, h]$, where $E$ is the number of edges.
3. *Context Embeddings $\phi_c$*: capture contextual information by transforming $m_c$ context features from the current decoding step $s_t$ from the feature space to the embedding space $h$, i.e., $[B, m_c] \rightarrow [B, h]$, for nodes or edges.

<div align="center">
<img src="https://github.com/ai4co/rl4co/assets/48984123/c47a9301-4c9f-43fd-b21f-761abeae9717"  style="width: 100%; height: auto;">
</div>


 Embeddings can be automatically selected by our library at runtime by simply passing the `env_name` to the policy. Additionally, we allow for granular control of any higher-level policy component independently, such as encoders and decoders.