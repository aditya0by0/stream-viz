from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from stream_viz.base import StrategyPlot


class LearningStrategyChart(StrategyPlot):
    """
    A class to plot learning strategies based on Kappa statistics.

    Parameters
    ----------
    kappa_df : pd.DataFrame
        DataFrame containing Kappa values for different strategies over time.
    """

    def __init__(self, kappa_df: pd.DataFrame):
        self._legend_handles: List = []
        self._processed_kappa_df, self._color_dict = self._compute_stats_for_graph(
            kappa_df
        )

    @classmethod
    def _compute_stats_for_graph(
        cls, kappa_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Compute statistics for graph plotting from the given Kappa DataFrame.

        Parameters
        ----------
        kappa_df : pd.DataFrame
            DataFrame containing Kappa values for different strategies over time.

        Returns
        -------
        Tuple[pd.DataFrame, dict]
            A tuple containing the processed DataFrame with additional statistics and a color dictionary for strategies.
        """
        _kappa_df = kappa_df.copy(deep=True)
        max_strategy = _kappa_df.idxmax(axis=1)

        sorted_kappa = _kappa_df.apply(lambda x: np.sort(x)[::-1], axis=1)
        _kappa_df["First_Kappa"] = sorted_kappa.apply(lambda x: x[0])
        _kappa_df["Second_Kappa"] = sorted_kappa.apply(lambda x: x[1])
        _kappa_df["Winner_Strategy"] = max_strategy

        unique_columns = kappa_df.columns
        color_dict = {
            strategy: plt.get_cmap("tab10")(i)
            for i, strategy in enumerate(unique_columns)
        }

        return _kappa_df, color_dict

    def plot(self, start_tpt: int, end_tpt: int) -> None:
        """
        Plot the learning strategy chart.

        Parameters
        ----------
        start_tpt : int
            Start timepoint for plotting.
        end_tpt : int
            End timepoint for plotting.
        """
        plt.figure(figsize=(18, 6))
        self.plot_winner_at_each_tpt(start_tpt, end_tpt)

        # Create a single legend outside the plot
        plt.legend(
            handles=self._legend_handles,
            title="Learning Strategy",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        # Adjust layout to make space for the legends
        plt.tight_layout(rect=(0.0, 0.0, 0.85, 1.0))
        plt.grid(True)
        plt.show()

    def plot_winner_at_each_tpt(
        self, start_batch: int, end_batch: int, step: int = 50
    ) -> None:
        """
        Plot the winner strategy at each timepoint within the specified range.

        Parameters
        ----------
        start_batch : int
            Start batch for plotting.
        end_batch : int
            End batch for plotting.
        step : int, optional
            Step size for the batches, by default 50.
        """
        self._check_time_batches(start_batch, end_batch)

        batch_list_window = self._processed_kappa_df.loc[
            start_batch:end_batch
        ].index.tolist()

        for i in range(len(batch_list_window) - 1):
            batch_i = batch_list_window[i]
            batch_next = batch_list_window[i + 1]
            x = [batch_i, batch_next]
            y = [
                self._processed_kappa_df["First_Kappa"].loc[batch_i],
                self._processed_kappa_df["First_Kappa"].loc[batch_next],
            ]
            _color = self._color_dict[
                self._processed_kappa_df["Winner_Strategy"].loc[batch_i]
            ]
            plt.plot(x, y, color=_color, marker="o")

            plt.fill_between(
                x,
                [
                    self._processed_kappa_df["First_Kappa"].loc[batch_i],
                    self._processed_kappa_df["First_Kappa"].loc[batch_next],
                ],
                [
                    self._processed_kappa_df["Second_Kappa"].loc[batch_i],
                    self._processed_kappa_df["Second_Kappa"].loc[batch_next],
                ],
                color=_color,
                alpha=0.3,
            )

        plt.xlabel("Batch")
        plt.ylabel("Kappa / Normalized difference within window")
        plt.title("Kappa Values with Color Indicating Strategy with Highest Score")

        legend_handles = [
            plt.Line2D([0], [0], color=self._color_dict[strategy], lw=2, label=strategy)
            for strategy in self._processed_kappa_df["Winner_Strategy"].unique()
        ]
        self._legend_handles += legend_handles

    def _check_time_batches(self, start_batch: int, end_batch: int) -> None:
        """
        Check if the specified time batches are valid.

        Parameters
        ----------
        start_batch : int
            Start batch for checking.
        end_batch : int
            End batch for checking.

        Raises
        ------
        ValueError
            If start or end batch is not in the data or out of range.
        """
        if start_batch not in self._processed_kappa_df.index:
            raise ValueError("Start batch not in data")

        if end_batch not in self._processed_kappa_df.index:
            raise ValueError("End batch not in data")

        if (
            start_batch < self._processed_kappa_df.index[0]
            or start_batch > self._processed_kappa_df.index[-1]
        ):
            raise ValueError("Check your start and/or end timepoints")


if __name__ == "__main__":
    from stream_viz.data_encoders.strategy_data_encoder import KappaStrategyDataEncoder
    from stream_viz.utils.constants import _LEARNING_STRATEGY_DATA_PATH

    # Instantiate and encode the Kappa strategy data
    kappa_encoder = KappaStrategyDataEncoder()
    kappa_encoder.read_csv_data(
        filepath_or_buffer=_LEARNING_STRATEGY_DATA_PATH,
        header=[0, 1, 2],
        index_col=[0, 1],
    )
    kappa_encoder.encode_data()

    # Create the learning strategy chart and plot it
    LearningStrategyChart(kappa_encoder.encoded_data).plot(
        start_tpt=11000, end_tpt=12950
    )
