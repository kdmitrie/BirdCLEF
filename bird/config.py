import os
import torch


class CFG:
    train_csv = '/kaggle/input/birdclef-2024/train_metadata.csv'
    sample_csv = '/kaggle/input/birdclef-2024/sample_submission.csv'
    min_duration = None
    max_duration = 30
    batch_size = 16

    FS = 32000
    DURATION = 5
    STEP = 1
    min_rating = 4.5
    num_classes = 182

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    num_epochs = 10
    interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Interactive'

    p = 0.5

    sg = {
        'window_size': 1024,
        'hop_size': 512,
        'fmin': 16,
        'fmax': 16386,
        'power': 2,
        'mel_bins': 128,
        'top_db': 80.0
    }

    simple_noiser = {
        'nocalls': '/kaggle/input/bc23-dataset-without-birds/nocalls.pkl',
        'spectra': '/kaggle/input/bc23-dataset-without-birds/noise_spectra.pkl',
        'duration': 30,
        'p': 0.25,
        'A': 1.0
    }

    froger = {
        'FS': 32000,
        'duration': 30,
        'frogs_path': '/kaggle/input/rain-forrest-frogs/frogs.pkl',
        'p': 0.25,
        'A': 1.0,
    }

    aug_params = {
        "num_masks_x": (2, 10),
        "num_masks_y": (2, 4),
        "mask_y_length": (1, 4),
        "mask_x_length": (10, 50),
        "fill_value": 0,
        "p": 0.3
    }

    noiser = {
        'voice_csv': '/kaggle/input/bc24-add-noise/hindi-speech-classification/dataset/train.csv',
        'voice_path': '/kaggle/input/bc24-add-noise/hindi-speech-classification/dataset/train',
        'voice_p': 0.1,
        'voice_n': 5,
        'voice_A': 1,
        'voice_air': {
            'min_distance': 200,
            'max_distance': 400,
            'p': 1,
        },

        'music_path': '/kaggle/input/bc24-add-noise/bc24-hindi-songs/india-songs.mp3',
        'music_p': 0.1,
        'music_n': 1,
        'music_A': 5,
        'music_air': {
            'min_distance': 3000,
            'max_distance': 10000,
            'p': 1,
        },

        'short_csv': '/kaggle/input/bc24-add-noise/noise-audio-data/ESC-50-master/meta/esc50.csv',
        'short_path': '/kaggle/input/bc24-add-noise/noise-audio-data/ESC-50-master/audio',
        'short_p': 0.1,
        'short_n': 4,
        'short_A': 10,
        'short_air': {
            'min_distance': 10,
            'max_distance': 1000,
            'p': 1,
        },

        'vehicle_csv': '/kaggle/input/bc24-add-noise/vehicle-type-sound-dataset/vehicle_type_sound_dataset/labels.csv',
        'vehicle_path': '/kaggle/input/bc24-add-noise/vehicle-type-sound-dataset/vehicle_type_sound_dataset',
        'vehicle_p': 0.1,
        'vehicle_n': 5,
        'vehicle_A': 1,
        'vehicle_air': {
            'min_distance': 10,
            'max_distance': 1000,
            'p': 1,
        },

        'ss_csv': '/kaggle/input/birdclef-2024/train_metadata.csv',
        'ss_path': '/kaggle/input/birdclef-2024/unlabeled_soundscapes',
    }
