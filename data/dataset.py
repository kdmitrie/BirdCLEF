import numpy as np
import psutil
from torch.utils.data import Dataset
from typing import Callable, Tuple, List
from .loader import DataProvider


class BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, data_provider: DataProvider, transform: Callable = None, shuffle=True, seed=42):
        """
            data_provider: the source of data items
            transform: additional augmentations
            shuffle: whether to shuffle the order of items
            seed: the seed number used for RNG
        """

        super().__init__()
        self.data_provider = data_provider
        self.transform = transform
        self.length = len(data_provider)

        self.shuffle = np.arange(len(data_provider))
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(self.shuffle)
        self.shuffle = self.shuffle.tolist()

    def train_test_split(self, train_size=0.8, test_size=0.2):
        thresh = int(len(self) * train_size / (train_size + test_size))
        ind1 = range(thresh)
        ind2 = range(thresh, len(self))
        train_ds = IndexedDataset(self, ind1)
        test_ds = IndexedDataset(self, ind2)
        return train_ds, test_ds

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tuple, np.ndarray]:
        index = self.shuffle[index]
        x, y = self.data_provider[index].as_xy()
        if self.transform:
            x = self.transform(x)
        return x, y


class IndexedDataset(Dataset):
    """Class used to select a part of the base dataset specified by indices array"""

    def __init__(self, base_dataset: Dataset, indices: List):
        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CachedDataset(Dataset):
    def __init__(self, base_dataset, limit_gb=28):
        self.ds = base_dataset
        self.limit = limit_gb * 1024 ** 3
        self.cache = {}
        self.proc = psutil.Process()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if index in self.cache.keys():
            return self.cache[index]

        item = self.ds[index]

        if self.proc.memory_info().rss < self.limit:
            self.cache[index] = item
        return item
