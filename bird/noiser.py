import audiomentations
import librosa
import numpy as np
import os
import pandas as pd
import torch

from .dataset import BirdDataset
from .config import CFG


class Noiser:
    def __init__(self, cfg):
        self.get_sg = BirdDataset(CFG).get_sg
        self.cfg = cfg

        self.voice_df = pd.read_csv(self.cfg['voice_csv'])
        self.vehicle_df = pd.read_csv(self.cfg['vehicle_csv'])
        self.ss_df = pd.read_csv(self.cfg['ss_csv'])

        short_categories = ['frog', 'car_horn', 'rain', 'crickets']
        self.short_df = pd.read_csv('/kaggle/input/noise-audio-data/ESC-50-master/meta/esc50.csv')
        self.short_df = self.short_df[self.short_df.category.isin(short_categories)]

        self.voice_air = audiomentations.AirAbsorption(**self.cfg['voice_air'])
        self.music_air = audiomentations.AirAbsorption(**self.cfg['music_air'])
        self.short_air = audiomentations.AirAbsorption(**self.cfg['short_air'])
        self.vehicle_air = audiomentations.AirAbsorption(**self.cfg['vehicle_air'])

    @staticmethod
    def read(fname, offset=None, duration=None):
        data, _ = librosa.load(fname, sr=CFG.FS, offset=offset, duration=duration)
        data = librosa.util.normalize(data)
        return data

    @staticmethod
    def fade(waveform, t=32000):
        win = np.ones_like(waveform)
        sin = np.sin(np.linspace(0, np.pi / 2, t))
        win[:t] = sin
        win[-t:] = sin[::-1]
        return waveform * win

    def random_voice(self, duration=30):
        path = self.cfg['voice_path']
        fname = f'{path}/{self.voice_df.sample().iloc[0].file_id}.mp3'
        data = self.read(fname)
        data = self.voice_air(data, sample_rate=CFG.FS)
        return data

    def random_music(self, duration=30):
        offset = np.random.randint(11000)
        data = self.read(self.cfg['music_path'], offset=offset, duration=duration)
        data = self.music_air(data, sample_rate=CFG.FS)
        return data

    def random_short(self, duration=30):
        path = self.cfg['short_path']
        fname = f'{path}/{self.short_df.sample().iloc[0].filename}'
        data = self.read(fname)
        data = self.short_air(data, sample_rate=CFG.FS)
        return data

    def random_vehicle(self, duration=30):
        path = self.cfg['vehicle_path']
        fname = f'{path}/{self.vehicle_df.sample().iloc[0].file_path}'
        data = self.read(fname)
        data = self.vehicle_air(data, sample_rate=CFG.FS)
        return data

    def random_ss(self, duration=30):
        offset = np.random.randint(240 - duration)
        path = self.cfg['ss_path']
        fname = os.listdir(self.cfg['ss_path'])[np.random.randint(8444)]

        data = self.read(f'{path}/{fname}', offset=offset, duration=duration)
        return data

    def random_sound(self, duration=30):
        ll = duration * CFG.FS
        waveform = np.zeros(ll)
        for what in ['music', 'voice', 'short', 'vehicle']:
            for _ in range(self.cfg[f'{what}_n']):
                if np.random.random() > self.cfg[f'{what}_p']:
                    continue

                amp = self.cfg[f'{what}_A'] * np.random.random()
                sound = amp * getattr(self, f'random_{what}')(duration=duration)
                sl = len(sound)
                offset = np.random.randint(ll - sl) if ll > sl else 0
                if what != 'music':
                    sound = self.fade(sound)
                waveform[offset: offset + sl] += sound
        return waveform

    def random_sg(self, duration=30):
        soundscape = self.random_ss(duration)
        soundscape = torch.Tensor(soundscape)
        soundscape_sg = self.sparse_sg(soundscape)

        noise = self.random_sound(duration)
        noise = torch.Tensor(noise)
        noise_sg = self.get_sg[0](noise).numpy()

        return soundscape_sg + noise_sg, soundscape_sg, noise_sg

    def sparse_sg(self, waveform, part=5e-4):
        sg0 = self.get_sg[0](waveform)
        sg1 = self.get_sg[1](sg0).numpy()

        sg_fft0 = np.fft.fft(sg0, axis=-1)
        sg_fft1 = np.fft.fft(sg1, axis=-1)

        a = abs(sg_fft1)
        k = int(sg_fft1.shape[0] * sg_fft1.shape[1] * (1 - part))
        ap = np.partition(a, k, axis=None)

        sg_fft0[a < ap[k]] = 0

        isg = np.fft.ifft(sg_fft0, axis=-1)
        isg = np.real(isg)
        isg = torch.Tensor(isg)
        return isg
