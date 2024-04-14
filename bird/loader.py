import numpy as np
import librosa
from dataclasses import dataclass

from ..data.loader import DataItem, DataReader
from .birdclef24 import BI


@dataclass
class BirdItem(DataItem):
    data: np.ndarray
    label: int
    fs: int

    def __repr__(self):
        return super().__repr__()


class BirdReader(DataReader):
    """Class for reading the train and test data"""
    fs: int = 32000

    def __init__(self, csv_file: str, ogg_path: str, limit: int = None):
        """
        csv_file: the dataframe
        ogg_path: the folder containing data in ogg format
        limit: manually limit the number of records
        """
        super().__init__(csv_file=csv_file, limit=limit)
        self.ogg_path = ogg_path

    def _get_one_item(self, item: object) -> DataItem:
        """Reads the data from ogg files"""
        label = BI()[item.primary_label]
        fname = f'{self.ogg_path}/{item.filename}'

        data, _ = librosa.load(fname, sr=self.fs)
        data = librosa.util.normalize(data)

        item = BirdItem(data=data, label=label, fs=self.fs)
        return self._process_one_item(item)
