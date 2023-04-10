Have a look at [ncobench/envs/tsp.py](../../ncobench/envs/tsp.py) for a deployed implementation.


---

## Experimental: `Memmap`

For now, this seems to be a bit slower and inefficient than just raw tensor loading. But it may become useful in the future.

```python
# Test

import torch
import torch.nn as nn

from tensordict import MemmapTensor
from tensordict.prototype import tensorclass
from torch.utils.data import DataLoader


@tensorclass
class TSPData:
    observations: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset, device=None):
        data = cls(
            observations=MemmapTensor(
                len(dataset), *dataset[1:].shape, dtype=torch.float32
            ),
            batch_size=[len(dataset)],
            device=device,
        )
        data = cls(observations=dataset, batch_size=[len(dataset)], device=device)
        return data
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = TSPEnv().generate_data(100000)

training_data_tc = TSPData.from_dataset(data, device=device)
```