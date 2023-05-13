import torch
import numpy as np
import colorednoise
from dataclasses import dataclass

@dataclass
class BirdAugmentation:
    """The class to add some augmentations to audio data"""
    sampling_rate:int = 320
    time_shift:float = 1.
    noise_level:float = 0.1


    def apply_time_shift(self, sig):
        shift = np.random.randint(int(self.sampling_rate * self.time_shift))
        return sig[..., shift:]


    def add_noise(self, sig):
        power = torch.sqrt(torch.sum(sig**2, dim=-1)/sig.shape[-1])
        degree = 0.5 + np.random.randn()
        noise = colorednoise.powerlaw_psd_gaussian(degree, sig.shape[-1])
        return sig + self.noise_level * power * noise


    def __call__(self, sig):
        sig = self.apply_time_shift(sig)
        sig = self.add_noise(sig)
        
        return sig
