import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from stream_viz.base import Binning


class DecisionTreeBinning(Binning):
    """
    Perform binning on numerical columns using a decision tree classifier.

    Parameters:
    ----------
    col_name_regex : str, optional
        Regular expression to select columns for binning, default is '.*'.
    bin_col_names : str, optional
        Prefix for binned columns, default is 'bin_'.

    Attributes:
    ----------
    _DT_model : DecisionTreeClassifier
        Decision tree model used for binning.
    column_mapping : dict
        Mapping of original columns to binned columns.
    bin_thresholds : list
        List of bin thresholds obtained from the decision tree model.
    binned_data_X : pd.DataFrame
        DataFrame containing binned numerical columns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._DT_model: DecisionTreeClassifier = DecisionTreeClassifier(
            criterion="entropy", max_depth=2
        )
        self.column_mapping = {}
        self.bin_thresholds = []
        self.binned_data_X = pd.DataFrame()

    def set_params_for_DT(self, *args, **kwargs):
        """
        Set parameters for the DecisionTreeClassifier.

        Parameters:
        ----------
        *args, **kwargs:
            Any arguments accepted by DecisionTreeClassifier.
        """
        self._DT_model = DecisionTreeClassifier(*args, **kwargs)

    def perform_binning(
        self, X_df_encoded_m: pd.DataFrame, y_encoded_m: pd.Series
    ) -> None:
        """
        Perform binning on numerical columns.

        Parameters:
        ----------
        X_df_encoded_m : pd.DataFrame
            Encoded dataframe containing numerical columns to bin.
        y_encoded_m : pd.Series
            Encoded target labels for training the decision tree model.
        """
        self._get_bins_using_DTs(X_df_encoded_m, y_encoded_m)
        _binned_data: pd.DataFrame = X_df_encoded_m.copy(deep=True)
        for col in X_df_encoded_m.filter(regex=self._col_name_regex).columns:
            binned_col_name = f"{self._bin_col_names}{col}"
            self.column_mapping[col] = binned_col_name  # Store the mapping
            _binned_data[binned_col_name] = X_df_encoded_m[col].apply(
                lambda x: self._apply_binning(x, self.bin_thresholds)
            )
        self.binned_data_X = _binned_data

    def _get_bins_using_DTs(self, X_df_encoded_m, y_encoded_m):
        """
        Train the decision tree model to get bin thresholds.

        Parameters:
        ----------
        X_df_encoded_m : pd.DataFrame
            Encoded dataframe containing numerical columns to bin.
        y_encoded_m : pd.Series
            Encoded target labels for training the decision tree model.
        """
        self._DT_model.fit(
            X_df_encoded_m.filter(regex=self._col_name_regex), y_encoded_m
        )

        # Extract the decision thresholds from the tree
        thresholds: np.ndarray = self._DT_model.tree_.threshold[
            self._DT_model.tree_.threshold > -2  # Filter out invalid thresholds
        ]
        self.bin_thresholds = thresholds.tolist()

    @staticmethod
    def _apply_binning(value, thresholds):
        """
        Apply binning based on given thresholds.

        Parameters:
        ----------
        value : numeric
            Value to bin.
        thresholds : list
            List of bin thresholds.

        Returns:
        -------
        int
            Bin index for the given value.
        """
        # Apply binning based on thresholds
        for i, thresh in enumerate(thresholds):
            if value <= thresh:
                return i
        return len(thresholds)  # If value exceeds the last threshold


if __name__ == "__main__":
    from stream_viz.data_encoders.cfpdss_data_encoder import NormalDataEncoder
    from stream_viz.utils.constants import _NORMAL_DATA_PATH

    # Cfpdss data encoding with missing values
    normal = NormalDataEncoder()
    normal.read_csv_data(
        filepath_or_buffer=_NORMAL_DATA_PATH,
    )
    normal.encode_data()

    dt_binner = DecisionTreeBinning()
    dt_binner.perform_binning(normal.X_encoded_data, normal.y_encoded_data)
    dt_binner.binned_data_X.head()
    print("Column Mapping: ", dt_binner.column_mapping)
