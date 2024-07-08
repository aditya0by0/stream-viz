from typing import List, Optional

import pandas as pd
from river import stream

from stream_viz.base import DriftDetector, Streamer
from stream_viz.feature_drift.f_drift_detector import FeatureDriftDetector
from stream_viz.real_drift.r_drift_detector import RealConceptDriftDetector


class DataStreamer(Streamer):

    def __init__(
        self,
        rcd_detector_obj: Optional[RealConceptDriftDetector] = None,
        fd_detector_obj: Optional[FeatureDriftDetector] = None,
    ):
        self.rcd_detector_obj: Optional[RealConceptDriftDetector] = rcd_detector_obj
        self.fd_detector_obj: Optional[FeatureDriftDetector] = fd_detector_obj
        self._drift_detectors: List[DriftDetector] = []

        if rcd_detector_obj is not None:
            self._drift_detectors.append(rcd_detector_obj)
        if fd_detector_obj is not None:
            self._drift_detectors.append(fd_detector_obj)

        if not self._drift_detectors:
            raise ValueError(
                "Atleast one of Drift Detector object need to provided,"
                "{RealConceptDriftDetector, FeatureDriftDetector}"
            )

    def stream_data(self, X_df: pd.DataFrame, y_df: pd.Series) -> None:
        timepoint: int = 0

        for x_i, y_i in stream.iter_pandas(X_df, y_df):
            # Update stats for real concept drift detector
            for drift_detector in self._drift_detectors:
                drift_detector.update(x_i, y_i, tpt=timepoint)
            timepoint += 1


if __name__ == "__main__":
    from stream_viz.data_encoders.cfpdss_data_encoder import (
        MissingDataEncoder,
        NormalDataEncoder,
    )
    from stream_viz.utils.constants import _MISSING_DATA_PATH, _NORMAL_DATA_PATH

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer=_MISSING_DATA_PATH,
        index_col=[0],
    )
    missing.encode_data()

    # normal = NormalDataEncoder()
    # normal.read_csv_data(_NORMAL_DATA_PATH)
    # normal.encode_data()
    #
    # # As the KS test is only for numerical features
    # X_numerical = normal.X_encoded_data[normal.original_numerical_cols]

    dt_streamer = DataStreamer(
        rcd_detector_obj=RealConceptDriftDetector(),
        fd_detector_obj=FeatureDriftDetector(missing.X_encoded_data.columns),
    )
    dt_streamer.stream_data(X_df=missing.X_encoded_data, y_df=missing.y_encoded_data)

    dt_streamer.rcd_detector_obj.plot(start_tpt=100, end_tpt=3000)
    dt_streamer.fd_detector_obj.plot(missing.X_encoded_data.columns[0])
