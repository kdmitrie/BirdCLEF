import numpy as np


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
