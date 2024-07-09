from abc import ABC
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import OneToOneFeatureMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from stream_viz.base import DataEncoder


class CfpdssDataEncoder(DataEncoder, ABC):
    """
    Base class for encoding CFpdss datasets.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()
        self._X_encoded_data: pd.DataFrame = pd.DataFrame()
        self._y_encoded_data: pd.Series = pd.Series()
        self.original_categorical_cols: List[str] = []
        self.original_numerical_cols: List[str] = []
        # self.encoded_categorical_cols: List[str] = []
        # self.encoded_numerical_cols: List[str] = []
        self.categorical_column_mapping: dict = {}
        # No change in numerical col names as of now, kept for future use and code consistency
        self.numerical_column_mapping: dict = {}

    @property
    def X_encoded_data(self) -> pd.DataFrame:
        """
        Returns the encoded features DataFrame.

        Returns
        -------
        pd.DataFrame
            Encoded features DataFrame.

        Raises
        ------
        ValueError
            If the encoded DataFrame is empty.
        """
        if self._X_encoded_data is None or self._X_encoded_data.empty:
            raise ValueError("Encoded Data is empty. Please call `encode_data` method")
        return self._X_encoded_data

    @X_encoded_data.setter
    def X_encoded_data(self, value: pd.DataFrame):
        """
        Sets the encoded features DataFrame.

        Parameters
        ----------
        value : pd.DataFrame
            Encoded features DataFrame.

        Raises
        ------
        ValueError
            If the value is not a pandas DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("encoded_data must be a pandas DataFrame")
        self._X_encoded_data = value

    @property
    def y_encoded_data(self) -> pd.Series:
        """
        Returns the encoded target Series.

        Returns
        -------
        pd.Series
            Encoded target Series.

        Raises
        ------
        ValueError
            If the encoded Series is empty.
        """
        if self._y_encoded_data is None or self._y_encoded_data.empty:
            raise ValueError(
                "encoded_data Data is empty. Please call `encode_data` method"
            )
        return self._y_encoded_data

    @y_encoded_data.setter
    def y_encoded_data(self, value: pd.Series):
        """
        Sets the encoded target Series.

        Parameters
        ----------
        value : pd.Series
            Encoded target Series.

        Raises
        ------
        ValueError
            If the value is not a pandas Series.
        """
        if not isinstance(value, pd.Series):
            raise ValueError("encoded_data must be a pandas Series")
        self._y_encoded_data = value

    def encode_data(self, feature_scaling: Optional[bool] = True, **kwargs) -> None:
        """
        Encodes the dataset into numerical format.

        Parameters
        ----------
        feature_scaling : Optional[bool], default=True
            Whether to apply feature scaling to numerical columns.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
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

        X_df_cat_one_hot, categorical_mapping = self._encode_categorical_data(
            X_df_categorical
        )
        self.categorical_column_mapping = categorical_mapping

        if feature_scaling:
            X_df_non_categorical = self._encode_numerical_data(X_df_non_categorical)

        # Note: No change in col. name, placeholder for future use and code consistency
        self.numerical_column_mapping = dict(
            zip(X_df_non_categorical.columns, X_df_non_categorical.columns)
        )

        # Concatenate categorical and numerical dataframes
        X_df_encoded = pd.concat([X_df_cat_one_hot, X_df_non_categorical], axis=1)

        # Encoding the target variable
        y_df = self.original_data[[target_variable_name]]
        y_encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        y_one_hot = y_encoder.fit_transform(y_df)
        y_encoded = pd.Series(y_one_hot.ravel())

        self.X_encoded_data = X_df_encoded
        self.y_encoded_data = y_encoded

    def _encode_categorical_data(
        self, X_df_categorical: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Encodes categorical data using one-hot encoding.

        Parameters
        ----------
        X_df_categorical : pd.DataFrame
            DataFrame containing categorical features.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            Encoded DataFrame and a dictionary mapping original columns to encoded columns.
        """
        encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        one_hot_encoded = encoder.fit_transform(X_df_categorical)
        encoded_columns = encoder.get_feature_names_out()
        X_df_cat_one_hot = pd.DataFrame(one_hot_encoded, columns=encoded_columns)
        # Create mapping from original columns to encoded columns
        categorical_mapping = dict(zip(X_df_categorical.columns, encoded_columns))
        return X_df_cat_one_hot, categorical_mapping

    def _encode_numerical_data(
        self, X_df_non_categorical: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Applies feature scaling to numerical data.

        Parameters
        ----------
        X_df_non_categorical : pd.DataFrame
            DataFrame containing numerical features.

        Returns
        -------
        pd.DataFrame
            Scaled numerical features DataFrame.
        """
        scaler = self._get_scaler()
        X_df_non_categorical = pd.DataFrame(
            scaler.fit_transform(X_df_non_categorical),
            columns=scaler.get_feature_names_out(),
        )
        return X_df_non_categorical

    def _get_scaler(self) -> Union[OneToOneFeatureMixin, TransformerMixin]:
        """
        Returns the scaler to be used for feature scaling.

        Returns
        -------
        Union[OneToOneFeatureMixin, TransformerMixin]
            Scaler instance.
        """
        return MinMaxScaler()

    @property
    def encoded_data(self) -> pd.DataFrame:
        """
        Raises an error since `encoded_data` is not used for Cfpdss dataset.

        Returns
        -------
        pd.DataFrame
            Not implemented.
        """
        raise NotImplementedError(
            "`encoded_data` is not used for Cfpdss dataset. "
            "Use `X_encoded` and `y_encoded` instead."
        )

    @encoded_data.setter
    def encoded_data(self, value: pd.DataFrame):
        """
        Raises an error since `encoded_data` is not used for Cfpdss dataset.

        Parameters
        ----------
        value : pd.DataFrame
            Not implemented.
        """
        raise NotImplementedError(
            "`encoded_data` is not used for Cfpdss dataset. "
            "Use `X_encoded` and `y_encoded` instead."
        )


class NormalDataEncoder(CfpdssDataEncoder):
    """
    Class for encoding CFpdss datasets without missing values.
    """

    pass


class MissingDataEncoder(CfpdssDataEncoder):
    """
    Class for encoding CFpdss datasets with missing values.
    """

    def _encode_categorical_data(
        self, X_df_categorical: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Encodes categorical data with handling for missing values.

        Parameters
        ----------
        X_df_categorical : pd.DataFrame
            DataFrame containing categorical features.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            Encoded DataFrame and a dictionary mapping original columns to encoded columns.
        """
        # Record the indices of missing values for each categorical feature
        missing_indices = {
            col: X_df_categorical[col].index[X_df_categorical[col].isna()].tolist()
            for col in self.original_categorical_cols
        }

        # Impute NaN values in categorical features with the most frequent value from the same column, as a temporary
        # solution for one hot encoding, inorder to not create an extra one-hot columns for nan values
        X_df_categorical = X_df_categorical.apply(
            lambda col: col.fillna(
                col.mode()[0] if not col.mode().empty else col.value_counts().index[0]
            )
        )

        # One hot encoding - Categorical data
        encoder = OneHotEncoder(sparse_output=False, drop="if_binary", dtype=np.int32)
        one_hot_encoded = encoder.fit_transform(X_df_categorical)
        encoded_columns = encoder.get_feature_names_out(self.original_categorical_cols)
        X_df_cat_one_hot = pd.DataFrame(
            one_hot_encoded, columns=encoded_columns, index=X_df_categorical.index
        )

        # Replace the imputed values back with NaNs in the encoded DataFrame
        for col, indices in missing_indices.items():
            for index in indices:
                cols_to_check = [
                    c for c in X_df_cat_one_hot.columns if c.startswith(col)
                ]
                for c in cols_to_check:
                    X_df_cat_one_hot.at[index, c] = np.nan

        # Create mapping from original columns to encoded columns
        categorical_mapping = dict(zip(X_df_categorical.columns, encoded_columns))

        return X_df_cat_one_hot, categorical_mapping


if __name__ == "__main__":
    from stream_viz.utils.constants import _MISSING_DATA_PATH, _NORMAL_DATA_PATH

    # Cfpdss data encoding without missing values
    normal = NormalDataEncoder()
    normal.read_csv_data(filepath_or_buffer=_NORMAL_DATA_PATH)
    normal.encode_data()
    print("Encoded categorical column mapping:")
    print(normal.categorical_column_mapping)
    normal.X_encoded_data.head()
    normal.y_encoded_data.head()

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer=_MISSING_DATA_PATH,
        index_col=[0],
    )
    missing.encode_data()
    print("Encoded categorical column mapping:")
    print(missing.categorical_column_mapping)
    missing.X_encoded_data.head()
    missing.y_encoded_data.head()
