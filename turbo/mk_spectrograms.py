import time

start_time = time.time()

import glob
import librosa
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
import numpy as np
import pickle
import sys
import torch

sys.path.append('/kaggle/input/bc24-lib')

from BirdCLEF.bird.dataset import BirdDataset
from BirdCLEF.bird.config import CFG


def get_sg(fname):
    data, _ = librosa.load(fname, sr=32000)
    data = librosa.util.normalize(data)
    if len(data) < 240 * 32000:
        data = np.pad(data, (0, 240 * 32000 - len(data)))

    data = torch.tensor(data)
    s_db = bd.get_sg(data)[None, None, ...]
    return s_db


prefix, start, stop = sys.argv[1:]
start = int(start)
stop = int(stop)

root = '/kaggle/input/birdclef-2024'
bd = BirdDataset(CFG)

with open('oggs.pkl', 'rb') as f:
    oggs = pickle.load(f)

shm = shared_memory.SharedMemory(name='spectrograms')
unregister(shm._name, 'shared_memory')
data = np.ndarray([len(oggs), 1, 128, 15001], buffer=shm.buf)

for n in range(start, stop):
    file_path = oggs[n]
    data[n] = get_sg(file_path)

with open(f'/tmp/{prefix}', 'w') as f:
    f.write('1')

shm.close()

print(f'--- {prefix} run in {time.time() - start_time} seconds ---')