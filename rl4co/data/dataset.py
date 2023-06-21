from typing import Any

import torch

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictDataset(Dataset):
    """Dataset compatible with TensorDicts
    For some reason, it is better to "disassemble" the TensorDict into a list of dicts
    We use a custom collate function to reassemble the TensorDicts
    NOTE: may want to make an issue on TorchRL to ask best TensorDict practices
    """

    def __init__(self, data):
        if isinstance(data, TensorDict):
            self.data = [
                {key: value[i] for key, value in data.items()}
                for i in range(data.shape[0])
            ]
        else:
            self.data = [d for d in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ExtraKeyDataset(Dataset):
    """Dataset that includes an extra key to add to the data dict
    This is useful for adding a REINFORCE baseline reward to the data dict
    We had extra_ to identify the key as an extra key
    """

    def __init__(self, dataset, extra):
        self.data = dataset.data
        self.extra = extra
        assert len(self.data) == len(self.extra), "Data and extra must be same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data["extra"] = self.extra[idx]
        return data


def tensordict_collate_fn(batch):
    """Collate function compatible with TensorDicts
    Reassemble the list of dicts into a TensorDict; seems to be way more efficient than using a TensorDictDataset
    https://github.com/pytorch-labs/tensordict/issues/374
    """
    if isinstance(batch[0], TensorDict):
        return torch.stack(batch)
    return TensorDict(
        {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
        batch_size=len(batch),
    )


class TensorDictCollate:
    def __init__(self) -> None:
        print(
            "Warning: TensorDictCollateFn is deprecated. Use tensordict_collate_fn instead."
        )
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return tensordict_collate_fn(*args, **kwds)
