from dataclasses import dataclass
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
