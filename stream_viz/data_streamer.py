from typing import Optional

import pandas as pd
from river import stream

from stream_viz.base import Streamer
from stream_viz.feature_drift.f_drift_detector import FeatureDriftDetector
from stream_viz.real_drift.r_drift_detector import RealConceptDriftDetector


class DataStreamer(Streamer):

    # def __init__(
    #     self,
    #     rcd_detector_obj: RealConceptDriftDetector,
    #     # fd_detector_obj: FeatureDriftDetector,
    # ):
    #     self.rcd_detector_obj: RealConceptDriftDetector = rcd_detector_obj
    #     # self.fd_detector_obj: FeatureDriftDetector = fd_detector_obj

    def stream_data(self, X_df: pd.DataFrame, y_df: pd.Series) -> None:
        timepoint: int = 0

        for x_i, y_i in stream.iter_pandas(X_df, y_df):
            # Update stats for real concept drift detector
            self.rcd_detector_obj.update(x_i, y_i, tpt=timepoint)

            # TODO : update stats for feature drift detector

            timepoint += 1


if __name__ == "__main__":
    from stream_viz.data_encoders.cfpdss_data_encoder import MissingDataEncoder
    from stream_viz.utils.constants import _MISSING_DATA_PATH

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer=_MISSING_DATA_PATH,
        index_col=[0],
    )
    missing.encode_data()

    dt_streamer = DataStreamer(rcd_detector_obj=RealConceptDriftDetector())
    dt_streamer.stream_data(X_df=missing.X_encoded_data, y_df=missing.y_encoded_data)

    dt_streamer.rcd_detector_obj.plot_drift(start_tpt=100, end_tpt=3000)
