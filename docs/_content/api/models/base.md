# Base Models

## Autoregressive Models

<img class="full-img" alt="policy" src="https://user-images.githubusercontent.com/48984123/281976545-ca88f159-d0b3-459e-8fd9-89799be9d1b0.png">

Autoregressive models are models that generate sequences one element at a time, where the probability of selecting the next element depends on the previous elements in the sequence - this can either be captured by the neural network itself or by _embeddings_ which modify the input to the neural network.


### Policy

```{eval-rst}
.. automodule:: rl4co.models.common.constructive.autoregressive.policy
   :members:
   :undoc-members:
```

### Encoder

```{eval-rst}
.. automodule:: rl4co.models.common.constructive.autoregressive.encoder
   :members:
   :undoc-members:
```

### Decoder

```{eval-rst}
.. automodule:: rl4co.models.common.constructive.autoregressive.decoder
   :members:
   :undoc-members:
```

## Nonautoregressive Models

Non-autoregressive models generate a heatmap of probabilities from one node to another (i.e. of size $N \times N$). They are faster to train and evaluate than autoregressive models, but they may need additional search on top to find good solutions.

### Policy

```{eval-rst}
.. automodule:: rl4co.models.common.nonautoregressive.policy
   :members:
   :undoc-members:
```

### Encoder

```{eval-rst}
.. automodule:: rl4co.models.common.nonautoregressive.encoder
   :members:
   :undoc-members:
```

### Decoder

Note that we still need a decoding class for the heatmap (for example, to mask out invalid actions).


```{eval-rst}
.. automodule:: rl4co.models.common.nonautoregressive.decoder
   :members:
   :undoc-members:
```

