from abc import ABC

import pandas as pd


class Base(ABC):
    def stream(self):
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


class DataEncoder(ABC):
    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def read_csv_data(self, *args, **kwargs) -> None:
        self.data = pd.read_csv(*args, **kwargs)
        if self.data is None or self.data.empty:
            raise "Unable to read data"

    def encode_data(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame):
        self._data = value


class Streamer(Base):
    pass
