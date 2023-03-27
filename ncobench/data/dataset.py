from torch.utils.data import Dataset


class TorchDictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] # note: use torch.stack to get batch