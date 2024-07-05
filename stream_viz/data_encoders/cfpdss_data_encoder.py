from abc import ABC
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from stream_viz.base import DataEncoder


class CfpdssData(DataEncoder, ABC):
    def __init__(self):
        super().__init__()
        self._X_encoded_data: pd.DataFrame = pd.DataFrame()
        self._y_encoded_data: pd.Series = pd.Series()
        self.original_categorical_cols: List[str] = []
        self.original_numerical_cols: List[str] = []
        self.encoded_categorical_cols: List[str] = []
        self.encoded_numerical_cols: List[str] = []

    @property
    def X_encoded_data(self) -> pd.DataFrame:
        if self._X_encoded_data is None or self._X_encoded_data.empty:
            raise ValueError("Encoded Data is empty. Please call `encode_data` method")
        return self._X_encoded_data

    @X_encoded_data.setter
    def X_encoded_data(self, value: pd.DataFrame):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("encoded_data must be a pandas DataFrame")
        self._X_encoded_data = value

    @property
    def y_encoded_data(self) -> pd.Series:
        if self._y_encoded_data is None or self._y_encoded_data.empty:
            raise ValueError(
                "encoded_data Data is empty. Please call `encode_data` method"
            )
        return self._y_encoded_data

    @y_encoded_data.setter
    def y_encoded_data(self, value: pd.Series):
        if not isinstance(value, pd.Series):
            raise ValueError("encoded_data must be a pandas Series")
        self._y_encoded_data = value

    def encode_data(self, feature_scaling: Optional[bool] = True, **kwargs) -> None:

        target_variable_name = kwargs.get("target_variable_name", "class")

        X_df = self.original_data.drop(columns=target_variable_name)

        # Separating categorical and non-categorical columns dataframes
        self.original_categorical_cols = X_df.select_dtypes(
            include=["object"]
        ).columns.tolist()
        X_df_categorical = X_df[self.original_categorical_cols]
        self.original_numerical_cols = [
            col for col in X_df.columns if col not in self.original_categorical_cols
        ]
        X_df_non_categorical = X_df[self.original_numerical_cols]

        X_df_cat_one_hot = self._encode_categorical_data(X_df_categorical)

        if feature_scaling:
            X_df_non_categorical = self._encode_numerical_data(X_df_non_categorical)

        # Concatenate categorical and numerical dataframes
        X_df_encoded = pd.concat(
            [
                X_df_cat_one_hot,
                X_df_non_categorical,
            ],
            axis=1,
        )

        # Encoding the target variable
        y_df = self.original_data[[target_variable_name]]
        y_encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        y_one_hot = y_encoder.fit_transform(y_df)
        y_encoded = pd.Series(y_one_hot.ravel())

        self.X_encoded_data = X_df_encoded
        self.y_encoded_data = y_encoded

    def _encode_categorical_data(self, X_df_categorical: pd.DataFrame) -> pd.DataFrame:
        # One hot encoding - Categorical dataframe
        encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        one_hot_encoded = encoder.fit_transform(X_df_categorical)
        self.encoded_categorical_cols = encoder.get_feature_names_out()
        X_df_cat_one_hot = pd.DataFrame(
            one_hot_encoded, columns=self.encoded_categorical_cols
        )
        return X_df_cat_one_hot

    def _encode_numerical_data(
        self, X_df_non_categorical: pd.DataFrame
    ) -> pd.DataFrame:
        # Feature scaling for numerical dataframe
        scaler = self._get_scaler()
        X_df_non_categorical = pd.DataFrame(
            scaler.fit_transform(X_df_non_categorical),
            columns=scaler.get_feature_names_out(),
        )
        return X_df_non_categorical

    def _get_scaler(self) -> OneToOneFeatureMixin | TransformerMixin:
        return MinMaxScaler()

    @property
    def encoded_data(self) -> pd.DataFrame:
        raise NotImplementedError(
            "`encoded_data`` is not used for Cfpdss dataset. "
            "Use `X_encoded` and `y_encoded` instead."
        )

    @encoded_data.setter
    def encoded_data(self, value: pd.DataFrame):
        raise NotImplementedError(
            "`encoded_data`` is not used for Cfpdss dataset. "
            "Use `X_encoded` and `y_encoded` instead."
        )


class NormalDataEncoder(CfpdssData):
    pass


class MissingDataEncoder(CfpdssData):

    def _encode_categorical_data(self, X_df_categorical: pd.DataFrame) -> pd.DataFrame:
        # Record the indices of missing values for each categorical feature
        missing_indices = {
            col: X_df_categorical[col].index[X_df_categorical[col].isna()].tolist()
            for col in self.original_categorical_cols
        }

        # Impute NaN values in categorical features with the most frequent value from the same column, as temporary
        # solution for one hot encoding, inorder to not create an extra one-hot columns for nan values
        X_df_categorical = X_df_categorical.apply(
            lambda col: col.fillna(
                col.mode()[0] if not col.mode().empty else col.value_counts().index[0]
            )
        )

        # One hot encoding - Categorical data
        encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        one_hot_encoded = encoder.fit_transform(X_df_categorical)
        self.encoded_categorical_cols = encoder.get_feature_names_out(
            self.original_categorical_cols
        )
        X_df_cat_one_hot = pd.DataFrame(
            one_hot_encoded,
            columns=self.encoded_categorical_cols,
            index=X_df_categorical.index,
        )

        # Replace the imputed values back with NaNs in the encoded DataFrame
        for col, indices in missing_indices.items():
            for index in indices:
                cols_to_check = [
                    c for c in X_df_cat_one_hot.columns if c.startswith(col)
                ]
                for c in cols_to_check:
                    X_df_cat_one_hot.at[index, c] = np.nan

        return X_df_cat_one_hot


if __name__ == "__main__":
    from stream_viz.utils.constants import _MISSING_DATA_PATH, _NORMAL_DATA_PATH

    # Cfpdss data encoding without missing values
    normal = NormalDataEncoder()
    normal.read_csv_data(filepath_or_buffer=_NORMAL_DATA_PATH)
    normal.encode_data()
    normal.X_encoded_data.head()
    normal.y_encoded_data.head()

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer=_MISSING_DATA_PATH,
        index_col=[0],
    )
    missing.encode_data()

    missing.X_encoded_data.head()
    missing.y_encoded_data.head()
