import sys

PREFIX, BACKBONE, T, CSV, PKL, EPOCHS = sys.argv[1:]
EPOCHS = int(EPOCHS)
T = int(T)

print(sys.argv[1:])

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import timm
import timm.scheduler
import torch

from BirdCLEF.bird.dataset import BirdDataset, MixedBirdDataset
from BirdCLEF.bird.config import CFG2
from BirdCLEF.models.transformer import BirdTransformer
from BirdCLEF.data.dataset import IndexedDataset
from dl.trainer import ModelTrainer, ModelAccuracy, ModelROCAUC, ModelMultilabelAccuracy


def seed_all(seed=42):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all()

CFG = CFG2

# %% [markdown]
# # Config

# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:30:28.183512Z","iopub.execute_input":"2024-06-11T17:30:28.183873Z","iopub.status.idle":"2024-06-11T17:30:28.189829Z","shell.execute_reply.started":"2024-06-11T17:30:28.183838Z","shell.execute_reply":"2024-06-11T17:30:28.188587Z"}}
CFG.min_duration = T
CFG.max_duration = T
CFG.interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Interactive'

# %% [markdown]
# # Dataset

# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:30:28.191210Z","iopub.execute_input":"2024-06-11T17:30:28.191679Z","iopub.status.idle":"2024-06-11T17:31:23.625074Z","shell.execute_reply.started":"2024-06-11T17:30:28.191640Z","shell.execute_reply":"2024-06-11T17:31:23.621866Z"}}
CFG.train_csv = CSV
with open(PKL, 'rb') as f:
    spectrograms = pickle.load(f)


# with open(f'/kaggle/input/bc24-data-t-{T}/first_and_last_{T}_sec_t_{T}_fs_32000.pkl', 'rb') as f:
#    spectrograms = pickle.load(f)

# with open(f'/kaggle/input/bc24-data-t-{T}/loop_{T}_sec_t_{T}_fs_32000.pkl', 'rb') as f:
#    spectrograms = pickle.load(f)

# CFG.train_csv = f'/kaggle/input/bc24-data-t-30/train_metadata.csv'
# with open('/kaggle/input/bc24-data-t-30/first_30_sec_t_30_fs_32000.pkl', 'rb') as f:
#    spectrograms = pickle.load(f)

# CFG.train_csv = f'/kaggle/input/bc24-data-t-30-loop/train_metadata.csv'
# with open('/kaggle/input/bc24-data-t-30-loop/loop_30_sec_t_30_fs_32000.pkl', 'rb') as f:
#    spectrograms = pickle.load(f)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:31:23.630196Z","iopub.execute_input":"2024-06-11T17:31:23.630799Z","iopub.status.idle":"2024-06-11T17:31:24.866006Z","shell.execute_reply.started":"2024-06-11T17:31:23.630740Z","shell.execute_reply":"2024-06-11T17:31:24.864820Z"}}
class BirdDataset_SG(BirdDataset):
    def __init__(self, cfg, get_label='onehot', distance=4000):
        super().__init__(cfg)

        # self.distance = distance
        # self.df = self.df[self.df.distance <= distance].copy()
        # self.init_weights()
        self.get_label = lambda item: self._get_label(item)

    def init_weights(self):
        vc = self.df.primary_label.value_counts()
        self.weights = max(vc) / vc
        self.min_distance = self.df.groupby('primary_label').distance.min()

    def _get_label(self, item):
        onehot = self.get_onehot_labels(item)
        # pl_weight = self.weights[item.primary_label]
        # d_weight = self.get_distance_weight(item)

        weight = 1  # pl_weight * d_weight
        return np.append(onehot, weight)

    def set_spectrograms(self, spectrograms):
        self.spectrograms = spectrograms

    def get_distance_weight(self, item):
        min_distance = self.min_distance[item.primary_label]
        min_distance = 0
        return 0.5 * (1 + np.cos(np.pi * (item.distance - min_distance) / self.distance))

    def __getitem__(self, index):
        item = self.df.iloc[index]
        sg = self.spectrograms[item.initial_index].astype(np.float32)
        sg = torch.tensor(sg)[None, ...]
        label = self.get_label(item)
        return sg, label


bd = BirdDataset_SG(CFG)
bd.set_spectrograms(spectrograms)

train_ds, val_ds = bd.train_test_split(part=0.001)
# train_ds = MixedBirdDataset(train_ds)

bd.dump_indices('bd_indices.pkl')

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

print(f'Num of birds = {len(bd.df.primary_label.unique())}; Num of records = {len(bd.df)}')

# Test the dataset
plt.figure(figsize=(24, 6))
plt.pcolormesh(bd[100][0].squeeze())
plt.show()


# %% [markdown]
# # Augmentations

# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:31:24.867598Z","iopub.execute_input":"2024-06-11T17:31:24.868006Z","iopub.status.idle":"2024-06-11T17:31:24.883091Z","shell.execute_reply.started":"2024-06-11T17:31:24.867969Z","shell.execute_reply":"2024-06-11T17:31:24.881941Z"}}
class Noiser():
    def __init__(self, cfg):
        self.cfg = cfg

        if 'nocalls' in cfg.keys():
            with open(cfg['nocalls'], 'rb') as f:
                self.nocalls = pickle.load(f)
        else:
            self.nocalls = None

        if 'spectra' in cfg.keys():
            with open(cfg['spectra'], 'rb') as f:
                self.spectra = pickle.load(f)
        else:
            self.spectra = None

    def get_nocall(self, waveform_len):
        indices = np.random.choice(range(len(self.nocalls)), size=np.ceil(waveform_len / 8 / CFG.FS).astype(int))
        noise = []
        for idx in indices:
            sig = self.nocalls[idx]
            sig /= np.std(sig)
            noise.append(sig)
        noise = np.concatenate(noise)

        if len(noise) > waveform_len:
            start = np.random.randint(len(noise) - waveform_len)
            noise = noise[start: start + waveform_len]

        return noise

    def get_spectra(self, sg_shape):
        z = np.zeros((1, sg_shape[-1]))
        idx = np.random.randint(len(self.spectra))
        z = z + self.spectra[idx][:, None]

        noise = np.random.normal(size=sg_shape[-2:]) ** 2 * z

        noise = np.log(noise)
        return noise

    def get_noise(self, dataset, sg):
        data_nc = self.get_nocall(waveform_len)

        data = data_nc

        S_dB = dataset.get_sg(data)[None, ...]

        return S_dB


# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:31:24.884513Z","iopub.execute_input":"2024-06-11T17:31:24.884878Z","iopub.status.idle":"2024-06-11T17:31:24.900698Z","shell.execute_reply.started":"2024-06-11T17:31:24.884849Z","shell.execute_reply":"2024-06-11T17:31:24.899295Z"}}
class Froger:
    def __init__(self, cfg):
        with open(cfg['frogs_path'], 'rb') as f:
            self.sounds = pickle.load(f)
            self.cfg = cfg

    def get_sound(self):
        waveform = np.array(())
        ll = self.cfg['duration'] * self.cfg['FS']
        while len(waveform) < ll:
            idx = np.random.randint(len(self.sounds))
            waveform = np.concatenate((waveform, self.sounds[idx]))
        return waveform[:ll]

    def get_sg(self, dataset):
        data = self.get_sound()
        S_dB = dataset.get_sg(torch.FloatTensor(data))[None, ...]
        return S_dB.numpy()


# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:34:55.951883Z","iopub.execute_input":"2024-06-11T17:34:55.952417Z","iopub.status.idle":"2024-06-11T17:34:55.959807Z","shell.execute_reply.started":"2024-06-11T17:34:55.952360Z","shell.execute_reply":"2024-06-11T17:34:55.958424Z"}}
class NoiseMixer:
    def __init__(self, path):
        self.data = np.load(path)

    def mix(self, x):
        ind = np.random.randint(len(self.data))
        noise = self.data[ind]
        shift = np.random.randint(noise.shape[-1])
        noise = np.roll(noise, shift, -1)
        a = np.random.uniform(low=0, high=1)
        return x + a * noise


# mixer = NoiseMixer(f'/kaggle/input/bc24-noise-generator-{T}sec-1-cfg2-result/noise_images_1.npy')


# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:34:56.858768Z","iopub.execute_input":"2024-06-11T17:34:56.859787Z","iopub.status.idle":"2024-06-11T17:35:00.039113Z","shell.execute_reply.started":"2024-06-11T17:34:56.859749Z","shell.execute_reply":"2024-06-11T17:35:00.037841Z"}}
aug_transforms = albumentations.Compose([
    albumentations.XYMasking(**CFG.aug_params),
    albumentations.RandomGridShuffle(grid=(6, 1), p=1),
])

CFG.noiser = {
    'nocalls': '/kaggle/input/bc23-dataset-without-birds-cfg2/nocalls.pkl',
    'spectra': '/kaggle/input/bc23-dataset-without-birds-cfg2/noise_spectra.pkl',
    'p': 0.25,
    'A': 1.0
}


class FCFG:
    froger = {
        'FS': CFG.FS,
        'duration': T,
        'frogs_path': '/kaggle/input/rain-forrest-frogs/frogs.pkl',
        'p': 0.25,
        'A': 1.0,
    }


noiser = Noiser(CFG.noiser)
froger = Froger(FCFG.froger)


def augmentations(batch, model):
    return batch

    result = []
    for img in batch:
        img = img[0].cpu().numpy()
        img = aug_transforms(image=img)['image'][None, ...]

        if (np.random.random() < CFG.noiser['p']):
            noise = noiser.get_spectra(img.shape)
            img = img + CFG.noiser['A'] * noise

        if (np.random.random() < FCFG.froger['p']):
            noise = froger.get_sg(bd)
            img = img + FCFG.froger['A'] * noise

        if (np.random.random() < 0.15):
            img_a = np.random.uniform(0.3, 1.2)
            img = mixer.mix(img_a * img)

        result.append(torch.FloatTensor(img))

    return torch.stack(result)


# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:35:00.040880Z","iopub.execute_input":"2024-06-11T17:35:00.041298Z","iopub.status.idle":"2024-06-11T17:35:00.048208Z","shell.execute_reply.started":"2024-06-11T17:35:00.041197Z","shell.execute_reply":"2024-06-11T17:35:00.047091Z"}}
# Test the augmentations
x1 = bd[100][0]
x2 = augmentations(x1[None, ...], None)

# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:35:00.049311Z","iopub.execute_input":"2024-06-11T17:35:00.049684Z","iopub.status.idle":"2024-06-11T17:35:00.916916Z","shell.execute_reply.started":"2024-06-11T17:35:00.049655Z","shell.execute_reply":"2024-06-11T17:35:00.915745Z"}}
fig, ax = plt.subplots(nrows=3)
fig.set_size_inches((24, 8))

g0 = ax[0].pcolormesh(x1.squeeze())
plt.colorbar(g0, ax=ax[0])

g1 = ax[1].pcolormesh(x2.squeeze())
plt.colorbar(g1, ax=ax[1])

g2 = ax[2].pcolormesh((x2 - x1).squeeze())
plt.colorbar(g2, ax=ax[2])

plt.show()

# %% [markdown]
# # Model

# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:35:00.919310Z","iopub.execute_input":"2024-06-11T17:35:00.919692Z","iopub.status.idle":"2024-06-11T17:35:00.931026Z","shell.execute_reply.started":"2024-06-11T17:35:00.919660Z","shell.execute_reply":"2024-06-11T17:35:00.929704Z"}}
from sklearn.metrics import roc_auc_score


class FocalLossWithWeight(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target, reduction='mean'):
        n = input.shape[-1]
        weights = target[:, -1:].expand(*target.shape[:-1], target.shape[-1] - 1)
        target = target[:, :-1]

        input = input.view(-1).float()
        target = target.reshape(-1).float()
        weights = weights.reshape(-1).float()

        loss = -target * torch.nn.functional.logsigmoid(input) * torch.exp(
            self.gamma * torch.nn.functional.logsigmoid(-input)) - \
               (1.0 - target) * torch.nn.functional.logsigmoid(-input) * torch.exp(
            self.gamma * torch.nn.functional.logsigmoid(input))

        wloss = loss * weights

        return n * wloss.mean() if reduction == 'mean' else wloss


class ModelROCAUCWithWeight(ModelROCAUC):
    def calc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        targets = targets[..., :-1]
        target_sums = targets.sum(axis=0)
        scored_columns = target_sums > 0
        return roc_auc_score(targets[:, scored_columns], predictions[:, scored_columns], average=self.average)


# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:35:00.932944Z","iopub.execute_input":"2024-06-11T17:35:00.933325Z","iopub.status.idle":"2024-06-11T17:35:00.948167Z","shell.execute_reply.started":"2024-06-11T17:35:00.933294Z","shell.execute_reply":"2024-06-11T17:35:00.946843Z"}}
def train_backbone(prefix, backbone, nn_model=None, epochs=30):
    if nn_model is None:
        print(f'\n\n!!! Creating new {backbone} model\n\n')
        nn_model = BirdTransformer(backbone=backbone, num_classes=CFG.num_classes)

    nn_model.augmentations = augmentations
    nn_model.to(CFG.device)
    nn_model.train()

    lr = 1e-4

    optimizer = torch.optim.Adam([
        {'params': nn_model.parameters(), 'lr': lr},
    ], lr=lr, weight_decay=1e-3, betas=(0.9, 0.999))
    scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=20, lr_min=1e-7)

    trainer = ModelTrainer(train_loader=train_loader,
                           val_loader=val_loader,
                           loss=FocalLossWithWeight(),
                           optimizer=optimizer,
                           scheduler=scheduler,
                           metrics=[ModelROCAUCWithWeight(), ModelMultilabelAccuracy()],
                           validation_calculate_rate=1,
                           limit_batches_per_epoch=0,
                           save_model_path=f'{prefix}_{backbone}_ep_%i.trch',
                           transfer_x_to_device=False,
                           switch_eval_train=True,
                           use_tqdm=True,
                           device=CFG.device)

    trainer(nn_model, epochs)


# %% [code] {"execution":{"iopub.status.busy":"2024-06-11T17:35:00.949799Z","iopub.execute_input":"2024-06-11T17:35:00.950250Z"}}
for n in range(3):
    seed_all(seed=n)
    train_backbone(f'{PREFIX}_seed={n}_t=t{T}', BACKBONE, epochs=EPOCHS)
