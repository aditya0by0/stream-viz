from typing import Type

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from stream_viz.base import Binning


class DecisionTreeBinning(Binning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._DT_model: Type[DecisionTreeClassifier] = DecisionTreeClassifier(
            criterion="entropy", max_depth=2
        )

    def set_params_for_DT(self, *args, **kwargs):
        self._DT_model = DecisionTreeClassifier(*args, **kwargs)

    def perform_binning(
        self, X_df_encoded_m: pd.DataFrame, y_encoded_m: pd.DataFrame
    ) -> None:
        # Perform binning for numerical columns
        self._get_bins_using_DTs(X_df_encoded_m, y_encoded_m)
        _binned_data: pd.DataFrame = X_df_encoded_m.copy(deep=True)
        for col in X_df_encoded_m.filter(regex=self._col_name_regex).columns:
            _binned_data[self._bin_col_names + col] = X_df_encoded_m[col].apply(
                lambda x: self._apply_binning(x, self.bin_thresholds)
            )
        self.binned_data_X = _binned_data

    def _get_bins_using_DTs(self, X_df_encoded_m, y_encoded_m):

        # Binning for numerical attributes
        self._DT_model.fit(
            X_df_encoded_m.filter(regex=self._col_name_regex), y_encoded_m
        )

        # Extract the decision thresholds from the tree
        thresholds: np.ndarray = self._DT_model.tree_.threshold[
            self._DT_model.tree_.threshold > -2
        ]
        self.bin_thresholds = thresholds.tolist()

    @staticmethod
    def _apply_binning(value, thresholds):
        # Define a function to apply binning based on thresholds
        # Filter out invalid thresholds
        for i, thresh in enumerate(thresholds):
            if value <= thresh:
                return i
        return len(thresholds)  # If value exceeds the last threshold


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

    dt_binner = DecisionTreeBinning()
    dt_binner.perform_binning(missing.X_encoded_data, missing.y_encoded_data)
    dt_binner.binned_data_X.head()
