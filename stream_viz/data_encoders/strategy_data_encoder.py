from typing import List, Tuple

import pandas as pd

from stream_viz.base import DataEncoder


class KappaStrategyDataEncoder(DataEncoder):
    __METRIC_NAME = "kappa"
    __BATCH_START_INDEX = "Batch_Start"
    __LEVEL_TWO_COLUMN = "summary"

    def __init__(self):
        super().__init__()
        self.strategy_columns: List = []

    def read_csv_data(self, *args, **kwargs) -> None:
        super().read_csv_data(*args, **kwargs)
        self.original_data = self.__preprocess_before_storing(self.original_data)

    def encode_data(self, *args, **kwargs):
        # No encoding needed for this type of data
        self.encoded_data = self.original_data

    @classmethod
    def __preprocess_before_storing(cls, df_experiment: pd.DataFrame) -> pd.DataFrame:
        """
        Removes multi-level index and multi-level columns from the dataframe, keeps only
        `Batch_Start` as the index and strategy as the only columns with values being the kappa score
        for given index and column.

        Parameters
        ----------
        df_experiment: pd.DataFrame
            Data related to Learning strategies / experiments dataframe

        Returns
        -------
        pd.DataFrame
            pre-processed dataframe
        """
        cls.strategy_columns = df_experiment.columns.get_level_values(0).unique()
        kappa_df = pd.DataFrame(columns=cls.strategy_columns)
        for learning_strategy in cls.strategy_columns:
            kappa_df[learning_strategy] = df_experiment[learning_strategy][
                cls.__LEVEL_TWO_COLUMN
            ][cls.__METRIC_NAME]
        kappa_df[cls.__BATCH_START_INDEX] = kappa_df.index.get_level_values(0)
        kappa_df = kappa_df.set_index(cls.__BATCH_START_INDEX)

        return kappa_df


if __name__ == "__main__":
    # Experiment data encoding
    kappa_encoder = KappaStrategyDataEncoder()
    kappa_encoder.read_csv_data(
        filepath_or_buffer="C:/Users/HP/Desktop/github-aditya0by0/stream-viz/data/experiment.csv",
        header=[0, 1, 2],
        index_col=[0, 1],
    )
    kappa_encoder.encoded_data()
    kappa_encoder.encoded_data.head()
