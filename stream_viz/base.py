from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

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


class Binning(ABC):
    def __init__(self, **kwargs):
        self._bin_thresholds: List[float] = []
        self._binned_data_X: pd.DataFrame = pd.DataFrame
        # regex pattern for the columns name that needs to binned
        self._col_name_regex = kwargs.get("col_name_regex", "^n")
        # New name for the columns to be binned
        self._bin_col_names = kwargs.get("bin_col_names", "_bin_idx_")

    @abstractmethod
    def perform_binning(self):
        pass

    @property
    def bin_thresholds(self) -> List[float]:
        if self._bin_thresholds is None:
            raise ValueError("bin_thresholds is empty")
        return self._bin_thresholds

    @bin_thresholds.setter
    def bin_thresholds(self, value: List[float]):
        if not isinstance(value, list):
            raise ValueError("bin_thresholds must be a List")
        self._bin_thresholds = value

    @property
    def binned_data_X(self) -> pd.DataFrame:
        if self._binned_data_X is None or self._binned_data_X.empty:
            raise ValueError("Binned Data is empty.")
        return self._binned_data_X

    @binned_data_X.setter
    def binned_data_X(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Input must be a Dataframe")
        self._binned_data_X = value
