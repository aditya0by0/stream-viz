from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class Base(ABC):
    def stream_data(self):
        raise NotImplementedError

    def update(self, x, idx):
        raise NotImplementedError

    def detect_concept_drift(self):
        raise NotImplementedError

    def plot_concept_drift(self):
        raise NotImplementedError

    def feature_drift(self):
        raise NotImplementedError

    def plot_feature_drift(self, feature):
        raise NotImplementedError

    def plot_velocity_numerical_var(self, feature):
        raise NotImplementedError

    def plot_velocity_categorical_var(self, feature):
        raise NotImplementedError

    def plot_missing_data(self):
        raise NotImplementedError

    def plot_learning_strategies(self):
        raise NotImplementedError


@dataclass
class DataEncoder(ABC):
    def __init__(self):
        self._original_data: pd.DataFrame = pd.DataFrame()
        self._encoded_data: pd.DataFrame = pd.DataFrame()

    def read_csv_data(self, *args, **kwargs) -> None:
        data = pd.read_csv(*args, **kwargs)
        if data is None or data.empty:
            raise ValueError("Unable to read data")
        self.original_data = data

    @abstractmethod  # Enforces that method must be implemented by any subclass
    def encode_data(self, *args, **kwargs):
        pass

    @property
    def original_data(self) -> pd.DataFrame:
        if self._original_data is None or self._original_data.empty:
            raise ValueError(
                "Original Data is empty. Please call `read_csv_data` method"
            )
        return self._original_data

    @original_data.setter
    def original_data(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("original_data must be a pandas DataFrame")
        self._original_data = value

    @property
    def encoded_data(self) -> pd.DataFrame:
        if self._encoded_data is None or self._encoded_data.empty:
            raise ValueError("Encoded Data is empty. Please call `encode_data` method")
        return self._encoded_data

    @encoded_data.setter
    def encoded_data(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("encoded_data must be a pandas DataFrame")
        self._encoded_data = value


class DriftDetector(Base):
    def __init__(self):
        pass

    def stream_data(self, data):
        pass
