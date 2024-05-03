import librosa
import numpy as np
import pandas as pd
import pickle
import re
import torch
import torchaudio
from typing import List

from ..data.dataset import IndexedDataset


class BirdDataset(torch.utils.data.Dataset):
    use_waveforms: bool = False

    def __init__(self, cfg, get_label='onehot'):
        self._df = pd.read_csv(cfg.train_csv)
        self.df = self._df
        self.df['initial_index'] = np.arange(len(self.df))

        self.cfg = cfg

        self.create_labels(cfg.sample_csv)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.FS,
            n_fft=cfg.sg['window_size'],
            win_length=cfg.sg['window_size'],
            hop_length=cfg.sg['hop_size'],
            f_min=cfg.sg['fmin'],
            f_max=cfg.sg['fmax'],
            pad=0,
            n_mels=cfg.sg['mel_bins'],
            power=cfg.sg['power'],
            normalized=False,
        )
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=self.cfg.sg['top_db'])

        self.get_sg = torch.nn.Sequential(mel_spec, amplitude_to_db)
        self.get_label = get_label

    def create_labels(self, sample_csv):
        df = pd.read_csv(sample_csv)
        self.labels = list(df.columns[1:])
        self.l2i = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def use_good(self, use_good=True, min_rating=5, min_records=500):
        if use_good:
            vc = self._df.primary_label.value_counts()
            self.df = self._df[
                self._df.primary_label.isin(vc[vc >= min_records].index) & (self._df.rating >= min_rating)].reset_index(
                drop=True)
        else:
            self.df = self._df

    def get_onehot_labels(self, item):
        secondary_labels = re.sub('\'|\[|\]| ', '', item.secondary_labels).split(',')
        onehot = np.zeros(self.cfg.num_classes)
        indices = [self.l2i[item.primary_label]]
        indices += [self.l2i[label] for label in secondary_labels if label in self.l2i.keys()]
        onehot[indices] = 1
        return onehot

    def get_primary_label(self, item):
        return self.l2i[item.primary_label]

    def set_waveforms(self, waveforms: List) -> None:
        self.waveforms = waveforms

    def __getitem__(self, index):
        item = self.df.iloc[index]

        if self.use_waveforms:
            data = self.waveforms[item.initial_index].astype(np.float32)
        else:
            data, _ = librosa.load(f'/kaggle/input/birdclef-2024/train_audio/{item.filename}', sr=self.cfg.FS)
            data = librosa.util.normalize(data)

        # 1. If data is to small, pad it to min_duration
        if (self.cfg.min_duration is not None) and (len(data) < self.cfg.min_duration * self.cfg.FS):
            data = np.pad(data, (0, self.cfg.min_duration * self.cfg.FS - len(data)))

        # 2. Cut the data to integer number of steps
        data_len = len(data) // (self.cfg.STEP * self.cfg.FS) * (self.cfg.STEP * self.cfg.FS)
        data = data[:data_len]

        # 3. Transform it into a tensor and into a SG
        data = torch.tensor(data)
        s_db = self.get_sg(data)[None, ...]

        if self.get_label == 'primary':
            label = self.get_primary_label(item)
        elif self.get_label == 'onehot':
            label = self.get_onehot_labels(item)
        elif callable(self.get_label):
            label = self.get_label(item)
        else:
            label = None

        return s_db, label

    def train_test_split(self, part=0.2, min_class_samples_train: int = 0, min_class_samples_test: int = 0):
        ll = len(self)
        self.indices = np.arange(0, ll)
        np.random.shuffle(self.indices)
        test_indices = self.indices[:int(part * ll)]
        train_indices = self.indices[int(part * ll):]

        test_indices = self._oversample(test_indices, min_class_samples_test)
        train_indices = self._oversample(train_indices, min_class_samples_train)

        return IndexedDataset(self, train_indices), IndexedDataset(self, test_indices)

    def _oversample(self, indices: np.ndarray, min_class_samples: int = 0) -> np.ndarray:
        local_df = self.df.iloc[indices]
        result = [indices]
        for label in local_df.primary_label.unique():
            chunk = local_df[local_df.primary_label == label].index.to_numpy()
            if len(chunk) < min_class_samples:
                additional = np.random.choice(chunk, min_class_samples - len(chunk))
                result.append(additional)
        return np.concatenate(result)

    def dump_indices(self, pkl):
        with open(pkl, 'wb') as f:
            pickle.dump(self.indices, f)


class MixedBirdDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, p=0.5):
        self.base_dataset = base_dataset
        self.p = p
        self.len = len(base_dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data1, label1 = self.base_dataset[idx]

        if np.random.random() < self.p:
            return data1, label1

        aux_idx = np.random.randint(self.len)
        data2, label2 = self.base_dataset[aux_idx]

        shift = np.random.randint(data2.shape[-1])
        data2 = np.roll(data2, shift)

        data = (data1 + data2) / 2
        label = np.clip(label1 + label2, a_min=0, a_max=1)

        return data, label
