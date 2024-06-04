import glob
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister
import numpy as np
import os
import pickle
import re


def create_memory_blocks(shm_name='spectrograms'):
    if len(glob.glob(f'/kaggle/input/birdclef-2024/test_soundscapes/*.ogg')) > 0:
        oggs = glob.glob('/kaggle/input/birdclef-2024/test_soundscapes/*.ogg')
    else:
        oggs = sorted(glob.glob(f'/kaggle/input/birdclef-2024/unlabeled_soundscapes/*.ogg'))[:1100]

    row_ids = []
    for n, file_path in enumerate(oggs):
        row_ids.append(re.search(r'/([^/]+)\.ogg$', file_path).group(1))

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.unlink()
    except:
        pass

    with open('oggs.pkl', 'wb') as f:
        pickle.dump(oggs, f)

    with open('row_ids.pkl', 'wb') as f:
        pickle.dump(row_ids, f)

    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=len(oggs) * 128 * 15001 * 8)
    unregister(shm._name, 'shared_memory')

    data = np.ndarray([len(oggs), 1, 128, 15001], buffer=shm.buf)

    return data, oggs, shm


def get_spectrograms(num_records, num_proc=4):
    block_len = int(np.ceil(num_records / num_proc))
    commands1, commands2 = [], []

    for n in range(num_proc):
        start = n * block_len
        stop = min(start + block_len, num_records)

        commands1.append(f'python mk_spectrograms.py sg_block_{n} {start} {stop} &')
        commands2.append(f'\nwhile ! test -f "/tmp/sg_block_{n}"; do\nsleep 1\ndone\n')

    with open('mk_spectrograms.sh', 'w') as f:
        f.write('rm /tmp/sg_block_*\n')
        f.write('\n'.join(commands1))
        f.write('\n'.join(commands2))

    os.system('bash mk_spectrograms.sh')
