import torch

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictDataset(Dataset):
    """Dataset compatible with TensorDicts.
    It is better to "disassemble" the TensorDict into a list of dicts.
    See :class:`tensordict_collate_fn` for more details.

    Note:
        Check out the issue on tensordict for more details:
        https://github.com/pytorch-labs/tensordict/issues/374.
        Note that directly indexing TensorDicts may be faster in creating the dataset
        but uses > 3x more CPU.
    """

    def __init__(self, data: TensorDict):
        self.data = [
            {key: value[i] for key, value in data.items()} for i in range(data.shape[0])
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tensordict_collate_fn(batch):
    """Collate function compatible with TensorDicts.
    Reassemble the list of dicts into a TensorDict; seems to be way more efficient than using a TensorDictDataset.

    Note:
        Check out the issue on tensordict for more details:
        https://github.com/pytorch-labs/tensordict/issues/374.
        Note that directly indexing TensorDicts may be faster in creating the dataset
        but uses > 3x more CPU.
    """
    return TensorDict(
        {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
        batch_size=len(batch),
    )


class ExtraKeyDataset(Dataset):
    """Dataset that includes an extra key to add to the data dict.
    This is useful for adding a REINFORCE baseline reward to the data dict.
    """

    def __init__(self, dataset: TensorDictDataset, extra: torch.Tensor):
        self.data = dataset.data
        self.extra = extra
        assert len(self.data) == len(self.extra), "Data and extra must be same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data["extra"] = self.extra[idx]
        return data
