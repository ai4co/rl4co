
import torch

from tensordict.tensordict import TensorDict
from torch.utils.data import Dataset


class TensorDictDataset(Dataset):
    """Dataset compatible with TensorDicts.
    Uses more CPU and has similar performance in loading to list comprehension, but is faster in instantiation
    than :class:`TensorDictDatasetList` (more than 10x faster).
    """

    def __init__(self, td: TensorDict):
        self.data = td

    def __len__(self):
        return len(self.data)

    def __getitems__(self, index):
        # Tricks:
        # - batched data loading with `__getitems__` for faster loading
        # - avoid directly indexing TensorDicts for faster loading
        return TensorDict(
            {key: item[index] for key, item in self.data.items()},
            batch_size=torch.Size([len(index)]),
            _run_checks=False,  # faster this way
        )

    def add_key(self, key, value):
        self.data.update({key: value})  # native method
        return self

def tensordict_collate_fn(x):
    """Equivalent to collating with `lambda x: x`"""
    return x


class TensorDictDatasetList(Dataset):
    """Dataset compatible with TensorDicts.
    It is better to "disassemble" the TensorDict into a list of dicts.
    See :class:`tensordict_collate_fn_list` for more details.

    Note:
        Check out the issue on tensordict for more details:
        https://github.com/pytorch-labs/tensordict/issues/374.
        Note that directly indexing TensorDicts may be faster in creating the dataset
        but uses > 3x more CPU.
    """

    def __init__(self, td: TensorDict):
        self.data_len = td.batch_size[0]
        self.data = [
            {key: value[i] for key, value in td.items()} for i in range(self.data_len)
        ]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.data[idx]

    def add_key(self, key, value):
        return ExtraKeyDataset(self, value, key_name=key)


def tensordict_collate_fn_list(batch):
    """Collate function compatible with TensorDicts that reassembles a list of dicts.

    Note:
        Check out the issue on tensordict for more details:
        https://github.com/pytorch-labs/tensordict/issues/374.
        Note that directly indexing TensorDicts may be faster in creating the dataset
        but uses > 3x more CPU.
    """
    return TensorDict(
        {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
        batch_size=torch.Size([len(batch)]),
        device=batch[0].device,
        _run_checks=False,
    )


class ExtraKeyDataset(TensorDictDatasetList):
    """Dataset that includes an extra key to add to the data dict.
    This is useful for adding a REINFORCE baseline reward to the data dict.
    Note that this is faster to instantiate than using list comprehension.
    """

    def __init__(
        self, dataset: TensorDictDatasetList, extra: torch.Tensor, key_name="extra"
    ):
        self.data_len = len(dataset)
        assert self.data_len == len(extra), "Data and extra must be same length"
        self.data = dataset.data
        self.extra = extra
        self.key_name = key_name

    def __getitem__(self, idx):
        data = self.data[idx]
        data[self.key_name] = self.extra[idx]
        return data
