import sys

PREFIX, BACKBONE, DURATION, EPOCHS = sys.argv[1:]
DURATION = int(DURATION)
EPOCHS = int(EPOCHS)

print(sys.argv[1:])

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys
import timm
import timm.scheduler
import torch

from BirdCLEF.bird.dataset import BirdDataset, MixedBirdDataset
from BirdCLEF.bird.config import CFG
from BirdCLEF.augmentations.simple_noiser import SimpleNoiser
from BirdCLEF.augmentations.froger import Froger
from BirdCLEF.augmentations.mixer import NoiseMixer

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

if 'torch_xla' not in sys.modules:
    print('Trying import torch_xla')

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.utils.utils as xu
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp


        class ModelTrainerXLA(ModelTrainer):

            def _backward_pass_train(self, model, loss_value):
                """Trains the model and returns an array of loss values"""

                # 1. We reset the gradient values ...
                self.optimizer.zero_grad()

                # 2. ... and calculate the new gradient values
                loss_value.backward()

                if self.max_clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_clip_grad_norm)

                # 3. Then we renew the model parameters
                self.optimizer.step()

                # 4. For TPU
                xm.mark_step()
                # or
                # xm.optimizer_step(self.optimizer, barrier=True)

                return loss_value.detach().cpu().numpy()


        ModelTrainer = ModelTrainerXLA
        CFG.device = xm.xla_device()

        print('\ttorch_xla successfuly imported')
    except ImportError:
        print('\ttorch_xla cannot be imported')

# # Config

CFG.min_duration = DURATION
CFG.simple_noiser['duration'] = CFG.min_duration
CFG.froger['duration'] = CFG.min_duration
CFG.train_csv = f'/kaggle/input/bc24-audio-loop-{CFG.min_duration}sec-dataset/train_metadata.csv'
spectrograms_pkl = f'/kaggle/input/bc24-audio-loop-{CFG.min_duration}sec-dataset/ALL_t_{CFG.min_duration}_fs_32000.pkl'
noise_mixer_npy = f'/kaggle/input/bc24-noise-generator-{CFG.min_duration}sec-1/noise_images_1.npy'

# %% [markdown]
# # Dataset

# %% [code] {"execution":{"iopub.status.busy":"2024-05-24T18:05:31.798851Z","iopub.execute_input":"2024-05-24T18:05:31.799278Z","iopub.status.idle":"2024-05-24T18:06:44.173875Z","shell.execute_reply.started":"2024-05-24T18:05:31.799244Z","shell.execute_reply":"2024-05-24T18:06:44.171499Z"}}
with open(spectrograms_pkl, 'rb') as f:
    spectrograms = pickle.load(f)


# %% [code] {"execution":{"iopub.status.busy":"2024-05-24T18:06:44.177900Z","iopub.execute_input":"2024-05-24T18:06:44.178647Z","iopub.status.idle":"2024-05-24T18:06:46.157684Z","shell.execute_reply.started":"2024-05-24T18:06:44.178611Z","shell.execute_reply":"2024-05-24T18:06:46.156161Z"}}
class BirdDataset_SG(BirdDataset):
    def __init__(self, cfg, get_label='onehot', distance=4000, no_birds=0):
        super().__init__(cfg)

        self.distance = distance

        self.no_birds = no_birds
        self.df = self.df[self.df.distance <= distance].copy()
        no_birds_df = pd.DataFrame([[-1, '', '', 0, '', 0, 0, 0, 0, 0, -1]] * no_birds, columns=self.df.columns)
        self.df = pd.concat((self.df, no_birds_df))

        self.init_weights()
        self.get_label = lambda item: self._get_label(item)

    def init_weights(self):
        vc = self.df.primary_label.value_counts()
        self.weights = max(vc) / vc
        self.min_distance = self.df.groupby('primary_label').distance.min()

    def _get_label(self, item):
        onehot = self.get_onehot_labels(item)
        pl_weight = self.weights[item.primary_label]
        d_weight = self.get_distance_weight(item)

        weight = pl_weight * d_weight
        return np.append(onehot, weight)

    def set_spectrograms(self, spectrograms):
        self.spectrograms = spectrograms

    def get_distance_weight(self, item):
        min_distance = self.min_distance[item.primary_label]
        min_distance = 0
        return 0.5 * (1 + np.cos(np.pi * (item.distance - min_distance) / self.distance))

    def nobird(self):
        sg = self.spectrograms[item.initial_index].astype(np.float32) * 0
        label = np.zeros(183)
        label[-1] = 1
        return sg, label

    def __getitem__(self, index):
        if index >= len(self.df) - self.no_birds:
            return self.nobird()

        item = self.df.iloc[index]
        sg = self.spectrograms[item.initial_index].astype(np.float32)
        sg = torch.tensor(sg)[None, ...]
        label = self.get_label(item)
        return sg, label


bd = BirdDataset_SG(CFG, no_birds=0)
bd.set_spectrograms(spectrograms)

train_ds, val_ds = bd.train_test_split()
train_ds = MixedBirdDataset(train_ds)

bd.dump_indices('bd_indices.pkl')

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)

print(f'Num of birds = {len(bd.df.primary_label.unique())}; Num of records = {len(bd.df)}')

# Test the dataset
plt.figure(figsize=(24, 6))
plt.pcolormesh(train_ds[100][0].squeeze())

# %% [markdown]
# # Augmentations

# %% [code] {"execution":{"iopub.status.busy":"2024-05-24T18:48:21.788873Z","iopub.execute_input":"2024-05-24T18:48:21.789422Z","iopub.status.idle":"2024-05-24T18:48:22.941440Z","shell.execute_reply.started":"2024-05-24T18:48:21.789381Z","shell.execute_reply":"2024-05-24T18:48:22.939443Z"}}
aug_transforms = albumentations.Compose([
    albumentations.XYMasking(**CFG.aug_params),
    albumentations.RandomGridShuffle(grid=(6, 1), p=1),
])

mixer = NoiseMixer(noise_mixer_npy)
noiser = SimpleNoiser(CFG.simple_noiser)
froger = Froger(CFG.froger)


def augmentations(batch, model):
    result = []
    for img in batch:
        img = img[0].cpu().numpy()
        img = aug_transforms(image=img)['image'][None, ...]

        if (np.random.random() < CFG.simple_noiser['p']):
            noise = noiser.get_spectra(img.shape)
            img = img + CFG.simple_noiser['A'] * noise

        if (np.random.random() < CFG.froger['p']):
            noise = froger.get_sg(bd)
            img = img + CFG.froger['A'] * noise

        if (np.random.random() < 0.15):
            img_a = np.random.uniform(0.3, 1.2)
            img = mixer.mix(img_a * img)

        result.append(torch.FloatTensor(img))

    return torch.stack(result)


# %% [code] {"execution":{"iopub.status.busy":"2024-05-24T18:50:13.998649Z","iopub.execute_input":"2024-05-24T18:50:13.999043Z","iopub.status.idle":"2024-05-24T18:50:15.710737Z","shell.execute_reply.started":"2024-05-24T18:50:13.999012Z","shell.execute_reply":"2024-05-24T18:50:15.709241Z"}}
# Test the augmentations

x1 = val_ds[100][0]
x2 = augmentations([x1], None)

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

# %% [code] {"execution":{"iopub.status.busy":"2024-05-24T18:50:18.780962Z","iopub.execute_input":"2024-05-24T18:50:18.781460Z","iopub.status.idle":"2024-05-24T18:50:18.796105Z","shell.execute_reply.started":"2024-05-24T18:50:18.781424Z","shell.execute_reply":"2024-05-24T18:50:18.795251Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2024-05-24T18:50:19.189958Z","iopub.execute_input":"2024-05-24T18:50:19.190749Z","iopub.status.idle":"2024-05-24T18:50:19.202851Z","shell.execute_reply.started":"2024-05-24T18:50:19.190704Z","shell.execute_reply":"2024-05-24T18:50:19.201318Z"}}
def train_backbone(backbone, nn_model=None):
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
                           save_model_path=f'{PREFIX}_{backbone}_epoch_%i.trch',
                           transfer_x_to_device=False,
                           switch_eval_train=True,
                           use_tqdm=CFG.interactive,
                           device=CFG.device)

    trainer(nn_model, EPOCHS)

train_backbone(BACKBONE, nn_model=None)
