from dataclasses import dataclass
import numpy as np
from .loader import DataItem, DataProcessor


@dataclass
class DataPipeline(DataProcessor):
    """Pipeline processing allows to apply several different data processors"""
    def __init__(self, *args):
        self.pipeline = args

    def process(self, item: DataItem) -> DataItem:
        for proc in self.pipeline:
            item = proc(item)
        return item


@dataclass
class ExpandDims(DataProcessor):
    """Adds new dimensions to the DataItem"""

    field: str
    dims: int = 0

    def process(self, item: DataItem) -> DataItem:
        data = getattr(item, self.field)
        data = np.expand_dims(data, self.dims)
        setattr(item, self.field, data)
        return item
