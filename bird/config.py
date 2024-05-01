import torch


class CFG:
    train_csv = '/kaggle/input/birdclef-2024/train_metadata.csv'
    sample_csv = '/kaggle/input/birdclef-2024/sample_submission.csv'
    max_duration = 30
    batch_size = 16

    FS = 32000
    min_rating = 4.5
    num_classes = 182

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    num_epochs = 10

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

    noiser = {
        'nocalls': '/kaggle/input/bc23-dataset-without-birds/nocalls.pkl',
        'spectra': '/kaggle/input/bc23-dataset-without-birds/noise_spectra.pkl',
        'p': 0.5,
        'A': 0.01
    }

    aug_params = {
        "num_masks_x": (2, 10),
        "num_masks_y": (2, 4),
        "mask_y_length": (1, 4),
        "mask_x_length": (10, 50),
        "fill_value": 0,
        "p": 0.3
    }
