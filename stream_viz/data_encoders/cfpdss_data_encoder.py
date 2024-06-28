from abc import ABC
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from stream_viz.base import DataEncoder


class CfpdssData(DataEncoder, ABC):
    def __init__(self):
        super().__init__()
        self.original_categorical_cols: List[str] = []
        self.original_numerical_cols: List[str] = []
        self.encoded_categorical_cols: List[str] = []
        self.encoded_numerical_cols: List[str] = []

    def encode_data(
        self, feature_scaling: Optional[bool] = True, **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.data.empty:
            raise "Data not read yet"

        target_variable_name = kwargs.get("target_variable_name", "class")

        X_df = self.data.drop(columns=target_variable_name)

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
        y_df = self.data[[target_variable_name]]
        y_encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        y_one_hot = y_encoder.fit_transform(y_df)
        y_encoded = pd.Series(y_one_hot.ravel())

        return X_df_encoded, y_encoded

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
    # Cfpdss data encoding without missing values
    normal = NormalDataEncoder()
    normal.read_csv_data(
        filepath_or_buffer="C:/Users/HP/Desktop/github-aditya0by0/stream-viz/data/cfpdss.csv"
    )
    X, y = normal.encode_data()
    X.head()

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer="C:/Users/HP/Desktop/github-aditya0by0/stream-viz/data/cfpdss_m0.5.csv"
    )
    X_missing, y = missing.encode_data()
    X_missing.head()
