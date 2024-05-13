import numpy as np
import pickle
from ..bird.config import CFG


class SimpleNoiser:
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

    def get_noise(self, dataset, waveform_len):
        data_nc = self.get_nocall(waveform_len)
        s_db = dataset.get_sg(data_nc)[None, ...]

        return s_db
