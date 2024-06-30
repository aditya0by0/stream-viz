from collections import deque
from typing import Dict, List, Tuple

import mplcursors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from rcd_configs import drift_detectors, metrics_dict, models_dict
from typing_extensions import Deque

from stream_viz.base import DriftDetector


class RealConceptDriftDetector(DriftDetector):
    def __init__(
        self,
        window_size=100,
        metric_name="Accuracy",
        model_name="Hoeffding",
        drift_detector="MDDM_A",
    ):
        super().__init__()
        self.concept_drifts_timepoints: List[int] = []
        self.warning_level_timepoints: List[int] = []
        self.metric_score_list: List[float] = []
        self.window_size = window_size
        self._metric_func = metrics_dict[metric_name]
        self._model = models_dict[model_name]
        self._drift_detector = drift_detectors[drift_detector](
            sliding_win_size=self.window_size,
            confidence=0.001,
            warning_confidence=0.005,
        )
        self._window_y: Deque[Tuple[float, float]] = deque(maxlen=self.window_size)

    def set_params_for_clf_model(self, *args, **kwargs):
        self._model = self._model.__class__(*args, **kwargs)

    def set_params_for_drift_dt(self, *args, **kwargs):
        self._drift_detector = self._drift_detector.__class__(*args, **kwargs)

    def update(self, x_i: Dict, y_i: int, tpt: int):
        y_pred = self._model.predict_one(x_i)
        if y_pred is None:
            y_pred = 0
        self._model.learn_one(x_i, y_i)

        self._window_y.append((y_i, y_pred))
        y_i_list, y_pred_list = zip(*self._window_y)
        win_metric_val = (
            self._metric_func(np.array(y_i_list), np.array(y_pred_list)) * 100
        )

        self.metric_score_list.append(win_metric_val)
        self.detect_drift(tpt, y_pred, y_i)

    def detect_drift(self, tpt: int, y_pred, y_i) -> None:
        self._drift_detector.input(int(y_pred == y_i))

        if self._drift_detector.is_warning_zone:
            self.warning_level_timepoints.append(tpt)
        if self._drift_detector.is_change_detected:
            self.concept_drifts_timepoints.append(tpt)
            self._model = self._model.clone()

    def plot_drift(
        self,
        start_tpt,
        end_tpt,
        name="None",
        vertical_line_height_percentage=100,
    ):
        plt.rcParams.update({"font.size": 15})
        plt.figure(1, figsize=(10, 6))
        sns.set_style("darkgrid")
        plt.clf()

        # Get the y-axis limit to calculate the height of vertical lines
        y_max = max(self.metric_score_list)
        vertical_line_height = (vertical_line_height_percentage / 100) * y_max

        # Plot warning levels with vertical lines
        for i, warning_lvl_timepoint in enumerate(self.warning_level_timepoints):
            if i == 0:  # Add label only for the first warning level
                plt.vlines(
                    warning_lvl_timepoint,
                    0,
                    vertical_line_height,
                    colors="orange",
                    linewidth=1,
                    linestyles="dashed",
                    label="Warning level for drift",
                )
            else:
                plt.vlines(
                    warning_lvl_timepoint,
                    0,
                    vertical_line_height,
                    colors="orange",
                    linewidth=0.5,
                    linestyles="dashed",
                    alpha=0.5,
                )

        # Plot concept drift with vertical lines
        for i, concept_drifts_timepoint in enumerate(self.concept_drifts_timepoints):
            if i == 0:  # Add label only for the first concept drift
                plt.vlines(
                    concept_drifts_timepoint,
                    0,
                    vertical_line_height,
                    colors="red",
                    linewidth=2,
                    linestyles="solid",
                    label="Drift detected",
                    alpha=0.8,
                )
            else:
                plt.vlines(
                    concept_drifts_timepoint,
                    0,
                    vertical_line_height,
                    colors="red",
                    linewidth=2,
                    linestyles="solid",
                    alpha=0.8,
                )

        # Plot metric score
        plt.plot(
            list(range(len(self.metric_score_list))),
            self.metric_score_list,
            "-b",
        )

        plt.legend(loc="best", facecolor="white", framealpha=1)
        plt.title(name + " on cfpdss dataset", fontsize=15)
        plt.xlabel("Timepoint")
        plt.ylabel("Metric Score")
        plt.xticks(np.arange(0, len(self.metric_score_list), 1000))

        mplcursors.cursor(hover=True)

        plt.show()


if __name__ == "__main__":
    from stream_viz.data_encoders.cfpdss_data_encoder import MissingDataEncoder

    # Cfpdss data encoding without missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer="C:/Users/HP/Desktop/github-aditya0by0/stream-viz/data/cfpdss_m0.5.csv",
        index_col=[0],
    )
    missing.encode_data()

    rl_cddt = RealConceptDriftDetector(window_size=100)
    # rl_cddt.stream_data(missing.X_encoded_data, missing.y_encoded_data)
    # rl_cddt.plot_concept_drift()
