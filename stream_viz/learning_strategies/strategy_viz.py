from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from stream_viz.base import StrategyPlot


class LearningStrategyChart(StrategyPlot):
    def __init__(self):
        self._legend_handles: List = []

    def plot_graph(self, kappa_df: pd.DataFrame, start_tpt, end_tpt):
        plt.figure(figsize=(18, 6))

        self.plot_winner_at_each_tpt(kappa_df, start_tpt, end_tpt)
        self.plot_differnce()

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

    def plot_winner_at_each_tpt(self, kappa_df: pd.DataFrame, start_tpt, end_tpt):
        # Determine the strategy with the highest kappa at each batch
        max_strategy = kappa_df.idxmax(axis=1)
        kappa_window_df = kappa_df.copy(deep=True)
        kappa_window_df["Max_Kappa"] = kappa_df.max(axis=1)
        kappa_window_df["Max_Strategy"] = max_strategy

        unique_columns = kappa_df.columns
        # Define colors for each strategy
        color_dict = {
            strategy: plt.cm.tab10(i) for i, strategy in enumerate(unique_columns)
        }

        # Plot the line with colors based on the strategy with the highest kappa
        kappa_window_df = kappa_window_df.loc[start_tpt:end_tpt]
        for i in range(len(kappa_window_df) - 1):
            x = [kappa_window_df.index[i], kappa_window_df.index[i + 1]]
            y = [
                kappa_window_df["Max_Kappa"].iloc[i],
                kappa_window_df["Max_Kappa"].iloc[i + 1],
            ]
            plt.plot(
                x,
                y,
                color=color_dict[kappa_window_df["Max_Strategy"].iloc[i]],
                marker="o",
            )

        # Add labels and title
        plt.xlabel("Batch")
        plt.ylabel("Kappa / Normalized differnce within window")
        plt.title("Kappa Values with Color Indicating Strategy with Highest Score")

        # Create legend handles for the strategies
        legend_handles = [
            plt.Line2D([0], [0], color=color_dict[strategy], lw=2, label=strategy)
            for strategy in unique_columns
        ]
        self._legend_handles += legend_handles

    def plot_difference(self):
        pass


if __name__ == "__main__":
    from stream_viz.data_encoders.strategy_data_encoder import KappaStrategyDataEncoder
    from stream_viz.utils.constants import _LEARNING_STRATEGY_DATA_PATH

    kappa_encoder = KappaStrategyDataEncoder()
    kappa_encoder.read_csv_data(
        filepath_or_buffer=_LEARNING_STRATEGY_DATA_PATH,
        header=[0, 1, 2],
        index_col=[0, 1],
    )
    kappa_encoder.encode_data()
    # kappa_encoder.encoded_data.head()
    LearningStrategyChart().plot_graph(
        kappa_encoder.encoded_data, start_tpt=11000, end_tpt=13000
    )
