## Constructive Policies

Constructive NCO policies pre-train a policy to amortize the inference. "Constructive" means that a solution is created from scratch by the model. We can also categorize constructive NCO in two sub-categories depending on the role of encoder and decoder:

#### Autoregressive (AR)
Autoregressive approaches **use a learned decoder** that outputs log probabilities for the current solution. These approaches generate a solution step by step, similar to e.g. LLMs. They have an encoder-decoder structure. Some models may not have an encoder at all and just re-encode at each step.

#### NonAutoregressive (NAR)
The difference between AR and NAR approaches is that NAR **only an encoder is learnable** (they just encode in one shot) and generate for example a heatmap, which can then be decoded simply by using it as a probability distribution or by using some search method on top.

Here is a general structure of a general constructive policy with an encoder-decoder structure:

<img class="full-img" alt="policy" src="https://user-images.githubusercontent.com/48984123/281976545-ca88f159-d0b3-459e-8fd9-89799be9d1b0.png">


where _embeddings_ transfer information from feature space to embedding space.

---



### Constructive Policy Base Classes

:::models.common.constructive.base



### Autoregressive Policies Base Classes

:::models.common.constructive.autoregressive.encoder

:::models.common.constructive.autoregressive.decoder

:::models.common.constructive.autoregressive.policy

### Nonautoregressive Policies Base Classes

:::models.common.constructive.nonautoregressive.encoder

:::models.common.constructive.nonautoregressive.decoder

:::models.common.constructive.nonautoregressive.policy
