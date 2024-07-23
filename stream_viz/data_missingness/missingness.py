from typing import Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyampute.exploration.mcar_statistical_tests import MCARTest
from scipy import stats

from stream_viz.base import InteractivePlot, Plotter
from stream_viz.data_encoders.cfpdss_data_encoder import (
    MissingDataEncoder,
    NormalDataEncoder,
)
from stream_viz.utils.binning import DecisionTreeBinning


class MarHeatMap(Plotter):
    """
    Class to compute and visualize the association between categorical columns and missing indicator columns
    using the chi-square test.

    Parameters:
    ----------
    normal_encoder_obj : NormalDataEncoder
        Object responsible for encoding normal data.
    missing_encoder_obj : MissingDataEncoder
        Object responsible for encoding missing data.
    na_col_name : str, optional
        Prefix for generated missing indicator columns. Default is 'is_na_'.
    """

    def __init__(
        self,
        normal_encoder_obj: NormalDataEncoder,
        missing_encoder_obj: MissingDataEncoder,
        na_col_name: str = "is_na_",
    ):
        self._na_col_name: str = na_col_name
        self._normal_encoder = normal_encoder_obj
        self._missing_encoder = missing_encoder_obj
        self._dt_binner_normal = DecisionTreeBinning()
        self._dt_binner_normal.perform_binning(
            self._normal_encoder.X_encoded_data, self._normal_encoder.y_encoded_data
        )

    def compute_mar_matrix(
        self, start_tpt: int = 200, end_tpt: int = 500
    ) -> pd.DataFrame:
        """
        Compute a matrix of p-values from chi-square tests between categorical and missing indicator columns.

        Parameters:
        ----------
        start_tpt : int, optional
            Starting time point for the data slice. Default is 200.
        end_tpt : int, optional
            Ending time point for the data slice. Default is 500.

        Returns:
        -------
        pd.DataFrame
            Matrix of p-values where rows correspond to categorical columns and columns to missing indicator columns.
        """
        col_list = list(
            self._normal_encoder.categorical_column_mapping.values()
        ) + list(self._dt_binner_normal.column_mapping.values())

        X_missing_indicator_df = self._get_missing_indicator_df()
        p_value_matrix = pd.DataFrame(
            index=col_list, columns=X_missing_indicator_df.columns
        )

        sliced_data_df = self._dt_binner_normal.binned_data_X[start_tpt:end_tpt]
        sliced_m_ind_df = X_missing_indicator_df[start_tpt:end_tpt]

        for data_col in col_list:
            for m_ind_col in X_missing_indicator_df.columns:
                # Create a contingency table
                contingency_table = pd.crosstab(
                    sliced_data_df[data_col],
                    sliced_m_ind_df[m_ind_col],
                )
                # Perform Chi-Square test
                chi2_stat, p_val, dof, ex = stats.chi2_contingency(contingency_table)
                p_value_matrix.loc[data_col, m_ind_col] = p_val

        return p_value_matrix.astype(float)

    def _custom_annotation_heatmap(
        self, p_value_matrix: pd.DataFrame, threshold: float
    ) -> np.ndarray:
        """
        Generate a custom annotation matrix for the heatmap based on a threshold.

        Parameters:
        ----------
        p_value_matrix : pd.DataFrame
            Matrix of p-values to be annotated.
        threshold : float
            Threshold below which values will not be annotated.

        Returns:
        -------
        np.ndarray
            Matrix of annotations where cells below the threshold are empty strings.
        """
        annot = np.full_like(
            p_value_matrix.values, "", dtype=object
        )  # Default to empty string
        for i in range(p_value_matrix.shape[0]):
            for j in range(p_value_matrix.shape[1]):
                val = p_value_matrix.iloc[i, j]
                if val > threshold:
                    annot[i, j] = f"{val:.3f}"  # Format as string
        return annot

    def plot(
        self, start_tpt: int = 200, end_tpt: int = 500, significance_level: float = 0.05
    ):
        """
        Plot a heatmap of p-values from chi-square tests with annotations highlighting significant associations.

        Parameters:
        ----------
        start_tpt : int, optional
            Starting time point for the data slice. Default is 200.
        end_tpt : int, optional
            Ending time point for the data slice. Default is 500.
        significance_level : float, optional
            Threshold for significance level. Values less than or equal to this will be highlighted. Default is 0.05.
        """
        p_value_matrix = self.compute_mar_matrix(start_tpt, end_tpt)
        annot = self._custom_annotation_heatmap(p_value_matrix, significance_level)

        # Create the heatmap
        plt.figure(figsize=(12, 8))
        heatmap = sns.heatmap(
            p_value_matrix,
            annot=True,  # annot, not using as of now due to standard error in seaborn
            cmap="coolwarm",
            cbar_kws={"label": "p-value"},
            linewidths=0.5,
            linecolor="black",
            annot_kws={"size": 10},
        )

        # Highlight cells with p-value <= significance_level with a different annotation
        for i in range(p_value_matrix.shape[0]):
            for j in range(p_value_matrix.shape[1]):
                p_val = p_value_matrix.iloc[i, j]
                if p_val <= significance_level:
                    plt.text(
                        j + 0.5,
                        i + 0.5,
                        f"{p_val:.3f}",  # f"$\\underline{{{p_val:.3f}}}$",  # Underlined LaTeX formatting
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="red",
                    )

        plt.title(
            f"Chi-Square Test p-values for data from {start_tpt} to {end_tpt} timepoint"
        )
        plt.show()

    def _get_missing_indicator_df(self) -> pd.DataFrame:
        """
        Create a DataFrame with missing indicator columns.

        Returns:
        -------
        pd.DataFrame
            DataFrame with columns indicating the presence of missing values.
        """
        X_missing_indicator_df = pd.DataFrame(
            columns=[
                self._na_col_name + str(col)
                for col in self._missing_encoder.X_encoded_data.columns
            ]
        )
        for col in self._missing_encoder.X_encoded_data.columns:
            X_missing_indicator_df[self._na_col_name + str(col)] = (
                self._missing_encoder.X_encoded_data[col].isna().astype(int)
            )
        return X_missing_indicator_df


class HeatmapPlotter(InteractivePlot, Plotter):
    """
    Plotter class for visualizing missing data as a heatmap.

    Parameters:
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data to plot.

    Methods:
    -------
    plot(start, end, features):
        Plot heatmap of missing values across specified features and time points.
    """

    def plot(self, start, end, features):
        """
        Plot heatmap of missing values.

        Parameters:
        ----------
        start : int
            Starting index for the slice of data to plot.
        end : int
            Ending index for the slice of data to plot.
        features : list
            List of features (columns) to include in the heatmap.
        """
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
    from stream_viz.data_encoders.cfpdss_data_encoder import (
        MissingDataEncoder,
        NormalDataEncoder,
    )
    from stream_viz.utils.constants import _MISSING_DATA_PATH, _NORMAL_DATA_PATH

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(
        filepath_or_buffer=_MISSING_DATA_PATH,
        index_col=[0],
    )
    missing.encode_data()

    # Cfpdss data encoding withOUT missing values
    normal = NormalDataEncoder()
    normal.read_csv_data(
        filepath_or_buffer=_NORMAL_DATA_PATH,
    )
    normal.encode_data()

    # ------------ Test Run : For MarHeatMap -----------------

    mar_hm = MarHeatMap(normal_encoder_obj=normal, missing_encoder_obj=missing)
    mar_hm.plot(start_tpt=200, end_tpt=500, significance_level=0.05)

    # ------------ Test Run : For HeatmapPlotter -----------------
    # plotter = HeatmapPlotter(missing.X_encoded_data)
    # plotter.display()
