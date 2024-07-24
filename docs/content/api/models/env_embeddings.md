# Environment Embeddings

In autoregressive policies, environment embeddings transfer data from feature space to hidden space:

- Initial Embeddings: encode global problem features
- Context Embeddings: modify current node embedding during decoding
- Dynamic Embeddings: modify all nodes embeddings during decoding

<img class="full-img" alt="policy" src="https://user-images.githubusercontent.com/48984123/281976545-ca88f159-d0b3-459e-8fd9-89799be9d1b0.png">


## Context Embeddings

The context embedding is used to modify the query embedding of the problem node of the current partial solution. Usually consists of a projection of gathered node embeddings and features to the embedding space.

:::models.nn.env_embeddings.context
    options:
      show_root_heading: false

## Dynamic Embeddings

The dynamic embedding is used to modify query, key and value vectors of the attention mechanism  based on the current state of the environment (which is changing during the rollout). Generally consists of a linear layer that projects the node features to the embedding space.

:::models.nn.env_embeddings.dynamic
    options:
      show_root_heading: false

## Init Embeddings

The init embedding is used to initialize the general embedding of the problem nodes without any solution information. Generally consists of a linear layer that projects the node features to the embedding space.

:::models.nn.env_embeddings.init
    options:
      show_root_heading: false