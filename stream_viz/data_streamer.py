import pandas as pd
from river import stream

from stream_viz.base import Streamer
from stream_viz.real_drift.r_drift_detector import RealConceptDriftDetector


class DataStreamer(Streamer):
    def stream_data(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> None:
        timepoint: int = 0

        rc_drift_detector = RealConceptDriftDetector()
        for x_i, y_i in stream.iter_pandas(X_df, y_df):
            # Update stats for real concept drift detector
            rc_drift_detector.update(x_i, y_i, tpt=timepoint)

            # TODO : update stats for feature drift detector

            timepoint += 1


if __name__ == "__main__":
    pass
    # DataStreamer().stream_data()
