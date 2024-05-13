import numpy as np
import pickle
import torch


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
        s_db = dataset.get_sg(torch.FloatTensor(data))[None, ...]
        return s_db.numpy()
