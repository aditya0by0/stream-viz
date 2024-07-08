from typing import Any

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy import stats

from stream_viz.base import InteractivePlot, Plotter
from stream_viz.utils.binning import DecisionTreeBinning


class MarHeatMap:
    def __init__(self, **kwargs):
        self._cat_col_regex: str = kwargs.get("cat_col_regex", r"^c\d*")
        self._bin_n_col_regex: str = kwargs.get("bin_n_col_regex", r"bin_idx*n*")
        self._na_col_regex: str = kwargs.get("na_col_regex", r"is_na_*\d*")
        self._na_col_name: str = kwargs.get("na_col_name", "is_na_")

    def compute_mar_matrix(self, X_df_encoded_m_ind: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = X_df_encoded_m_ind.filter(
            regex=self._cat_col_regex
        ).columns
        binned_numerical_columns = X_df_encoded_m_ind.filter(
            regex=self._bin_n_col_regex
        ).columns
        col_list = list(categorical_columns) + list(binned_numerical_columns)

        is_na_columns = X_df_encoded_m_ind.filter(regex=self._na_col_regex).columns

        p_value_matrix = pd.DataFrame(index=col_list, columns=is_na_columns)

        for col1 in col_list:
            for col2 in is_na_columns:
                # Create a contingency table
                contingency_table = pd.crosstab(
                    X_df_encoded_m_ind[col1], X_df_encoded_m_ind[col2]
                )
                # Perform Chi-Square test
                chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)
                p_value_matrix.loc[col1, col2] = p_val

        return p_value_matrix.astype(float)

    def plot_graph(
        self, X_df_encoded_m_ind, start_tpt=200, end_tpt=500, significance_level=0.05
    ):
        # Create a mask for p-values <= 0.05
        X_data = X_df_encoded_m_ind.iloc[start_tpt:end_tpt]
        X_data = self._add_null_indicator_cols(X_data)
        p_value_matrix = self.compute_mar_matrix(X_data)

        # Create a heatmap with highlighting
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(
            p_value_matrix,
            annot=True,
            cmap="coolwarm",
            cbar_kws={"label": "p-value"},
            linewidths=0.5,
            linecolor="black",
            annot_kws={"size": 10},
        )

        # Highlight cells with p-value <= 0.05 with a different annotation
        for i in range(p_value_matrix.shape[0]):
            for j in range(p_value_matrix.shape[1]):
                p_val = p_value_matrix.iloc[i, j]
                if p_val <= significance_level:
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        f"{p_val:.3f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                        weight="bold",
                        color="red",
                        fontstyle="italic",
                    )
                else:
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        f"{p_val:.3f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                    )

        plt.title(f"Chi-Square Test p-values")
        plt.show()

    def _add_null_indicator_cols(self, X_df_encoded_m):
        X_df_encoded_m_ind = X_df_encoded_m.copy(deep=True)
        for col in X_df_encoded_m.filter(regex="^(?!.*bin_idx)").columns:
            X_df_encoded_m_ind[self._na_col_name + col] = (
                X_df_encoded_m_ind[col].isna().astype(int)
            )
        return X_df_encoded_m_ind


class HeatmapPlotter(InteractivePlot, Plotter):

    def plot(self, start, end, features):
        plt.figure(figsize=(15, 6))
        selected_df = self._data_df.iloc[start:end][list(features)]
        ax = sns.heatmap(selected_df.isnull(), cmap="viridis", cbar=False)
        plt.xlabel("Attributes")
        plt.ylabel("Time Points")

        purple = mpatches.Patch(color="purple", label="not missing")
        yellow = mpatches.Patch(color="yellow", label="missing")
        plt.legend(handles=[purple, yellow], loc="upper right")

        # Create depth_list directly with desired values
        depth_list = np.arange(start, end, 1000)  # Adjust this as needed

        # Calculate y-ticks to match the number of rows in the dataframe
        yticks = np.linspace(0, selected_df.shape[0] - 1, len(depth_list), dtype=int)

        # Set y-ticks and y-tick labels
        ax.set_yticks(yticks)
        ax.set_yticklabels(depth_list)

        plt.show()

    def _add_interactive_plot(self):
        super()._add_interactive_plot()


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

    # ------------ Test Run : For Mar Heat Map -----------------
    # dt_binner = DecisionTreeBinning()
    # dt_binner.perform_binning(missing.X_encoded_data, missing.y_encoded_data)
    #
    # mar_hm = MarHeatMap()
    # mar_hm.plot_graph(
    #     dt_binner.binned_data_X, start_tpt=200, end_tpt=500, significance_level=0.05
    # )

    # ------------ Test Run : For HeatmapPlotter -----------------
    plotter = HeatmapPlotter(missing.X_encoded_data)
    plotter.display()
