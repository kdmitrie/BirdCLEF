import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import List, Union, Callable, Optional
from abc import ABC, abstractmethod
from pathlib import Path


@dataclass
class DataItem:
    """Base class, that represents a data item"""

    def __repr__(self) -> str:
        shapes = []
        values = []
        for attr_name in dir(self):
            # Select only ordinary attributes of the object
            attr = getattr(self, attr_name)
            if attr_name[0] != '_' and not callable(attr):
                # Print shape
                if isinstance(attr, np.ndarray):
                    shape_str = f'{attr_name}: shape = {attr.shape}'
                else:
                    shape_str = f'{attr_name}: {attr.__class__}'
                shapes.append(shape_str)

                # Print data
                values.append(f'{attr_name}: ' + attr.__repr__())

        if len(shapes) and len(values):
            shapes.append('-' * 80)

        return '\n'.join(shapes + values)


@dataclass
class DataProvider:
    """Base class that represents a data provider.
    It contains a list of items, that are accessible by id and
    a list of methods, which are applied to the items on the fly"""
    limit: int = None
    verbose: bool = True
    process: List[Callable] = field(default_factory=list)

    def __len__(self) -> int:
        return 0

    def __getitem__(self, item: object) -> DataItem:
        return DataItem()

    def print(self, s: str) -> None:
        if self.verbose:
            print(s)

    def add_processing(self, process: Callable) -> None:
        self.process.append(process)

    def _process_one_item(self, item: DataItem) -> DataItem:
        for proc in self.process:
            item = proc(item)
        return item


class DataReader(DataProvider, ABC):
    """Class for reading the data.
    This class is a data provider, which uses the initial data as its input"""

    def __init__(self, csv_file: str, limit: int = None):
        """
        csv_file: the dataframe
        limit: manually limit the number of records
        """
        self.df = pd.read_csv(csv_file)
        self.limit = limit
        self.process = []

    def __len__(self) -> int:
        return len(self.df) if self.limit is None else self.limit

    def __getitem__(self, selection: Union[int, slice]) -> Union[DataItem, List[DataItem]]:
        if isinstance(selection, int):
            return self._get_one_item(self.df.iloc[selection, :])
        return [self._get_one_item(item) for _, item in self.df.iloc[selection, :].iterrows()]

    @abstractmethod
    def _get_one_item(self, item: object) -> DataItem:
        pass


@dataclass
class DataProcessor(ABC):
    """The base class for data processing. It can process individual items or
    list of them or even all data providers as a whole"""

    def __call__(self, data: Union[DataItem, List[DataItem], DataProvider]) \
            -> Union[DataItem, List[DataItem], DataProvider]:
        if isinstance(data, DataItem):
            return self.process(data)

        if isinstance(data, DataProvider):
            data.add_processing(self.process)
            return data

        return [self.process(item) for item in data]

    @abstractmethod
    def process(self, item: DataItem) -> DataItem:
        return item


@dataclass
class DataSave:
    """Class for saving the data after some processing"""
    source: DataReader = None
    processor: DataProcessor = None
    path: str = './output'
    chunk_size: int = 1000
    verbose: bool = True

    def print(self, s: str) -> None:
        if self.verbose:
            print(s)

    def save(self) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)

        if self.source is None:
            raise ValueError('Source must not be None')

        for n in range(0, len(self.source), self.chunk_size):
            m = min(n + self.chunk_size, len(self.source))
            self.print(f'Items {n} - {m}')

            self.print(f'\tloading')
            items = self.source[n:m]

            if self.processor is not None:
                self.print(f'\tprocessing')
                items = self.processor(items)

            self.print(f'\tsaving')
            with open(f'{self.path}/{n}.pkl', 'wb') as f:
                pickle.dump(items, f)
