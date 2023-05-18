import numpy as np
import pandas as pd
from os import path
import pickle
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from typing import Callable

class BirdSplitDataset(Dataset):
    """Class used to split dataset into train/validation"""
    def __init__(self, base_dataset, start=0, stop=-1):
        super().__init__()
        self.base_dataset = base_dataset
        self.start = start
        self.stop = stop if stop > 0 else len(base_dataset)
        
    def __len__(self):
        return self.stop - self.start
    
    def __getitem__(self, idx):
        return self.base_dataset[self.start + idx]
    
    
class BirdSampleDataset(Dataset):
    """Class used to sample dataset from the base dataset according to the given probabilities array"""
    def __init__(self, base_dataset, probabilities, size=-1, min_probability=0.01, seed=42):
        super().__init__()
        self.base_dataset = base_dataset
        self.size = len(base_dataset) if size == -1 else size
        self.probabilities = probabilities
        self.min_probability = min_probability
        
        labels = base_dataset.get_primary_labels()
        item_select_probability = self.get_item_select_probability(labels)

        rng = np.random.default_rng(seed=seed)
        self.index = rng.choice(range(len(base_dataset)), size=self.size, p=item_select_probability)


    def get_item_select_probability(self, labels):
        item_select_probability = np.clip(1 - self.probabilities[labels], self.min_probability, self.size)
        return item_select_probability / np.sum(item_select_probability)
        
        
    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return self.base_dataset[self.index[idx]]


    
class BirdTrainDataset(Dataset):
    """Base class for handling BirdCLEF competition"""
    limit : None = None
    offset = 0
    multiple_dataset_len = 2000
    multiple_dataset_sample_rate = 320

    def __init__(self, csv: str, path: str, duration:int=0, max_duration:int=0, transform:Callable=None, shuffle=True, seed=42):
        """
            csv: path to csv file
            path: path to data files
            duration: the length of intervals the whole record is splitted to, in seconds. If less or equal to zero, no splitting is performed
            max_duration: the total length of record in seconds. If less or equal to zero, not restricted
            transform: additional augmentations
            shuffle: whether to shuffle the order of items
            seed: the seed number used for RNG
        """
        super().__init__()
        self.df = pd.read_csv(csv)
        self.path = path
        self.transform = transform
        self._create_dictionary()
        self.duration = duration
        self.max_duration = max_duration
        
        self.shuffle = np.arange(len(self.df))
        if shuffle:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(self.shuffle)
        
        # If % is found inside path, it is considered multiple-dataset
        self._get_data = self._get_data_multiple if ('%' in path) else self._get_data_single
            
            
    def get_primary_label(self, index):
        return self.bird2index[self.df.loc[index].primary_label]

        
    def get_primary_labels(self):
        return [self.get_primary_label(index) for index in self.shuffle]

        
    def get_label(self, index):
        return self.get_primary_label(index)

        
    def set_limit(self, limit, offset=0):
        self.limit = limit
        self.offset = offset


    def _index_transform(self, idx):
        return self.shuffle[idx + self.offset]
    
    
    def _pick_random_interval(self, data, sample_rate):
        """If data is long enough, pick random interval of `duration` length; else pad the signal with zeros"""
        samples = sample_rate * self.duration
        pad = samples - data.shape[-1]
        if pad > 0:
            return torch.nn.functional.pad(data, (pad // 2, pad - pad // 2))
        offset = torch.randint(high=-pad, size=(1,))
        return data[..., offset : offset + samples]
        
        
    def _create_dictionary(self):
        self.index2bird = sorted(self.df['primary_label'].unique())
        self.bird2index = dict(zip(self.index2bird, range(len(self.index2bird))))
        
        
    def train_test_split(self, train_size=0.8, test_size=0.2):
        thresh = int(len(self) * train_size / (train_size + test_size))
        train_ds = BirdSplitDataset(self, 0, thresh)
        test_ds = BirdSplitDataset(self, thresh)
        return train_ds, test_ds
    
    
    def __len__(self):
        if self.limit is None:
            return len(self.df)
        return self.limit
    
    
    def _get_data_single(self, primary_label, index):
        fname = path.join(self.path, self.df.loc[index].filename)
        data, sample_rate = torchaudio.load(fname)
        return data, sample_rate
        
    
    def _get_data_multiple(self, primary_label, index):
        dataset_num = 1 + index // self.multiple_dataset_len
        fname = path.join(self.path % dataset_num, 'data', str(primary_label), str(index))
        with open(fname, 'rb') as f:
            data = pickle.load(f)
            sample_rate = self.multiple_dataset_sample_rate
        return data, sample_rate
    
    
    def __getitem__(self, index):
        index = self._index_transform(index)
        primary_label = self.get_primary_label(index)
        label = self.get_label(index)
        
        data, sample_rate = self._get_data(primary_label, index)
        
        if self.max_duration > 0:
            data = data[..., : sample_rate*self.max_duration]
        
        if self.duration > 0:
            data = self._pick_random_interval(data, sample_rate)
        
        if self.transform:
            data = self.transform(data)
        
        return data, label


    
class MultiBirdTrainDataset(BirdTrainDataset):
    def get_secondary_labels(self, index):
        secondary_birds = json.loads(self.df.iloc[index].secondary_labels.replace("'", '"'))
        return [self.bird2index[bird] for bird in secondary_birds]
            
            
    def get_label(self, index):
        labels = np.zeros(len(self.index2bird))
        primary = self.get_primary_label(index)
        secondary =self.get_secondary_labels(index)
        labels[primary] = 1
        labels[secondary] = 1
        return labels
