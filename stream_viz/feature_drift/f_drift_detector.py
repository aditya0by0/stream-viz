from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

from stream_viz.base import DriftDetector
from stream_viz.utils.drifts_types import FeatureDriftType, get_fd_drift_type_keys


class FeatureDriftDetector(DriftDetector):
    def __init__(self, features_list: List[str], window_size=100, ks_test_pval=0.001):
        self._drift_records: List[Dict[str, str]] = []
        self._valid_keys: set[str] = get_fd_drift_type_keys()
        self.window_size: int = window_size
        self._window: Deque[Dict[str, float]] = deque(maxlen=window_size)
        self._drift_timepoints: List[int] = []
        self._moving_avg: pd.DataFrame = pd.DataFrame(columns=features_list)
        self._ks_test_pval: float = ks_test_pval
        self._drift_tp_df: pd.DataFrame = pd.DataFrame(columns=features_list)
        self._feature_data_df: pd.DataFrame = pd.DataFrame(columns=features_list)

    def update(self, x_i: Dict[str, float], y_i: int, tpt: int):
        self._window.append(x_i)
        self._feature_data_df.loc[tpt] = x_i

        if len(self._window) == self.window_size:
            self.detect_drift(tpt)

    def detect_drift(self, tpt: int):
        window_df = pd.DataFrame(self._window)
        for feature in window_df.columns:
            drift_detected, drift_type = self._detect_drift_using_ks(
                window_df[feature].values, self.window_size, self._ks_test_pval
            )
            if drift_detected:
                self._drift_tp_df.loc[tpt, feature] = drift_type

            self._moving_avg.loc[tpt, feature] = window_df[feature].mean()

    @staticmethod
    def _detect_drift_using_ks(
        window_data: np.ndarray, win_size: int, p_val: float
    ) -> Tuple[bool, Optional[str]]:
        first_half = window_data[: win_size // 2]
        second_half = window_data[win_size // 2 :]

        ks_stat, p_value = ks_2samp(first_half, second_half)
        if p_value < p_val:
            mean_diff = np.mean(second_half) - np.mean(first_half)
            if np.abs(mean_diff) > np.std(window_data):
                return True, "sudden_drift"
            elif mean_diff > 0:
                return True, "linear_drift"
            else:
                return True, "gradual_drift"

        return False, None

    def plot_drift(self, feature_name, window_size=None):
        if window_size is None:
            window_size = self.window_size
        feature_data = self._feature_data_df[feature_name]
        plt.figure(figsize=(10, 6))
        plt.scatter(feature_data.index, feature_data, marker="o", s=2)

        moving_mean = feature_data.rolling(window=window_size).mean()
        plt.plot(
            feature_data.index,
            moving_mean,
            color="black",
            linestyle="-",
            label=f"{feature_name} Moving Mean",
        )

        # drift_points, drift_types, moving_avg = self.detect_feature_drift(
        #     feature_data, window_size, 3
        # )

        # plt.plot(moving_avg.index, moving_avg, color='orange', linestyle='-', label=f'{feature} Moving Mean with trimming')

        drift_type_temp_label = []
        for idx, drift_type in self._drift_tp_df[feature_name].items():
            color = (
                "red"
                if drift_type == "Sudden Drift"
                else "orange" if drift_type == "Linear Drift" else "blue"
            )
            plt.axvline(
                x=idx,
                color=color,
                linestyle="--",
                label=(
                    f"{drift_type}" if drift_type not in drift_type_temp_label else ""
                ),
            )
            drift_type_temp_label.append(drift_type)

        plt.title(f"{feature_name} vs. Time")
        plt.xlabel("Time")
        plt.ylabel(f"{feature_name}")
        plt.grid(True)
        plt.xticks(np.arange(0, len(feature_data), 1000))
        plt.legend()
        plt.show()

    @property
    def drift_records(self) -> List[FeatureDriftType]:
        return self._drift_records

    @drift_records.setter
    def drift_records(self, drift_record: FeatureDriftType):
        if isinstance(drift_record, dict) and self._validate_drift(drift_record):
            self._drift_records.append(drift_record)
        else:
            raise ValueError("Invalid drift record")


if __name__ == "__main__":
    from stream_viz.data_encoders.cfpdss_data_encoder import (
        MissingDataEncoder,
        NormalDataEncoder,
    )
    from stream_viz.data_streamer import DataStreamer
    from stream_viz.utils.constants import _MISSING_DATA_PATH, _NORMAL_DATA_PATH

    # Cfpdss data encoding with missing values
    # missing = MissingDataEncoder()
    # missing.read_csv_data(
    #     filepath_or_buffer=_MISSING_DATA_PATH,
    #     index_col=[0],
    # )
    normal = NormalDataEncoder()
    normal.read_csv_data(_NORMAL_DATA_PATH)
    normal.encode_data()

    # As the KS test is only for numerical features
    X_numerical = normal.X_encoded_data[normal.original_numerical_cols]

    dt_streamer = DataStreamer(
        fd_detector_obj=FeatureDriftDetector(X_numerical.columns)
    )
    dt_streamer.stream_data(X_df=X_numerical, y_df=normal.y_encoded_data)

    dt_streamer.fd_detector_obj.plot_drift(feature_name=X_numerical.columns[0])

    # ----- Test: Feature Drift Detection for numerical variables on Dummy drift data -----
    # features_list = ["n_feature_1", "n_feature_2"]
    # drift_detector = FeatureDriftDetector(
    #     features_list=features_list, window_size=100, ks_test_pval=0.001
    # )
    #
    # # Generate data for 3 distributions for each feature
    # random_state = np.random.RandomState(seed=42)
    # dist_a_f1 = random_state.normal(0.8, 0.05, 1000)
    # dist_b_f1 = random_state.normal(0.4, 0.02, 1000)
    # dist_c_f1 = random_state.normal(0.6, 0.1, 1000)
    #
    # dist_a_f2 = random_state.normal(0.3, 0.04, 1000)
    # dist_b_f2 = random_state.normal(0.7, 0.03, 1000)
    # dist_c_f2 = random_state.normal(0.5, 0.05, 1000)
    #
    # # Concatenate data to simulate a data stream with 2 drifts for each feature
    # stream_f1 = np.concatenate((dist_a_f1, dist_b_f1, dist_c_f1))
    # stream_f2 = np.concatenate((dist_a_f2, dist_b_f2, dist_c_f2))
    #
    # # Simulate streaming data update
    # for i, (val_f1, val_f2) in enumerate(zip(stream_f1, stream_f2)):
    #     x_i = {"n_feature_1": val_f1, "n_feature_2": val_f2}
    #     drift_detector.update(x_i, 1, i)
    #
    # drift_detector._drift_tp_df.head()
    # drift_detector._moving_avg_df.head()
