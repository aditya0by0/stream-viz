from abc import ABC


class Base(ABC):
    def stream(self):
        raise NotImplementedError

    def read_data_from_file(self):
        raise NotImplementedError

    def encode_data(self, data):
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
    def read_data(self):
        raise NotImplementedError

    def encode_data(self):
        raise NotImplementedError

    def _custom_preprocessing(self):
        raise NotImplementedError


class Streamer(Base):
    pass
