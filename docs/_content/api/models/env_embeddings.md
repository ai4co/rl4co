# Environment Embeddings

## Context Embeddings

The context embedding is used to modify the query embedding of the problem node of the current partial solution. Usually consists of a projection of gathered node embeddings and features to the embedding space.


```{eval-rst}
.. automodule:: rl4co.models.nn.env_embeddings.context
   :members:
   :undoc-members:
```

---

## Dynamic Embeddings

The dynamic embedding is used to modify query, key and value vectors of the attention mechanism  based on the current state of the environment (which is changing during the rollout). Generally consists of a linear layer that projects the node features to the embedding space.

```{eval-rst}
.. automodule:: rl4co.models.nn.env_embeddings.dynamic
   :members:
   :undoc-members:
```

---

## Init Embeddings

The init embedding is used to initialize the general embedding of the problem nodes without any solution information. Generally consists of a linear layer that projects the node features to the embedding space.

```{eval-rst}
.. automodule:: rl4co.models.nn.env_embeddings.init
   :members:
   :undoc-members:
```