import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tensordict.tensordict import TensorDict


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


class TensorDictCollate(nn.Module):
    """Collate function compatible with TensorDicts
    Reassemble the list of dicts into a TensorDict; seems to be way more efficient than using a TensorDictDataset
    Why you ask? No idea, but it works. May want to make an issue on TorchRL to ask best TensorDict practices
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        if isinstance(batch[0], TensorDict):
            return torch.stack(batch)
        return TensorDict(
            {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys()},
            batch_size=len(batch),
        )
