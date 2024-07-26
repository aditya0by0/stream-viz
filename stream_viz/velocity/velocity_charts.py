import itertools
from typing import Iterable, List, Union

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from stream_viz.base import Velocity
from stream_viz.data_encoders.cfpdss_data_encoder import CfpdssDataEncoder


class StackedBarChart(Velocity):
    @staticmethod
    def plot(
        df: pd.DataFrame,
        feature: str,
        chunk_size: int,
        start_period: int,
        end_period: int,
        x_label_every: int = 5,
    ) -> None:
        """
        Plots a stacked bar chart for a given categorical feature over specified time periods.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data to plot.
        feature : str
            The name of the categorical feature to plot.
        chunk_size : int
            The number of data points in each chunk.
        start_period : int
            The starting period for the plot.
        end_period : int
            The ending period for the plot.
        x_label_every : int, optional
            Interval for displaying x-axis labels, by default 5.

        Returns
        -------
        None
        """
        feature = feature.strip()

        # Calculate the number of chunks
        num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

        # Create a column for the chunk number
        df["chunk"] = (df.index // chunk_size).astype(int)

        # Group by the chunk and calculate the counts for each value
        counts_df = (
            df.groupby("chunk")[feature]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={0: f"{feature}_0", 1: f"{feature}_1"})
        )
        counts_df[f"{feature}_Missing"] = df.groupby("chunk")[feature].apply(
            lambda x: x.isna().sum()
        )

        # Filter the combined_df to include only the specified range
        counts_df = counts_df.iloc[start_period:end_period]

        ind = np.arange(start_period, end_period)  # the x locations for the groups
        width = 1.0  # the width of the bars

        fig, ax = plt.subplots(figsize=(18, 6))

        # Plot for feature
        bar_0 = ax.bar(
            ind,
            counts_df[f"{feature}_0"],
            width,
            label=f"{feature} 0",
            color="blue",
            edgecolor="black",
        )
        bar_1 = ax.bar(
            ind,
            counts_df[f"{feature}_1"],
            width,
            bottom=counts_df[f"{feature}_0"],
            label=f"{feature} 1",
            color="orange",
            edgecolor="black",
        )
        bar_nan = ax.bar(
            ind,
            counts_df[f"{feature}_Missing"],
            width,
            bottom=counts_df[f"{feature}_0"] + counts_df[f"{feature}_1"],
            label=f"{feature} Missing",
            color="green",
            edgecolor="black",
        )

        # Generate x-labels showing the range of time points for each chunk
        chunk_labels = []
        for chunk in range(start_period, end_period):
            start = chunk * chunk_size + 1
            end = (chunk + 1) * chunk_size
            chunk_labels.append(f"{start}-{end}")

        # Set labels for only every 5th chunk
        tick_locs = np.arange(start_period, end_period)
        tick_labels = [
            label if (i % x_label_every == 0) else ""
            for i, label in enumerate(chunk_labels)
        ]

        ax.set_xlabel("Time period")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Stacked Bar Graph of {feature} for each time period of {chunk_size} Instances (Periods {start_period} to {end_period})"
        )
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.show()


class RollingMeansStds(Velocity):
    @staticmethod
    def plot(
        df: pd.DataFrame,
        features: List[str],
        window_size: int = 10,
        start_tp: int = 200,
        end_tp: int = 500,
    ) -> None:
        """
        Plots rolling means and standard deviations for given numerical features.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data to plot.
        features : List[str]
            List of numerical feature names to plot.
        window_size : int, optional
            The size of the moving window, by default 10.
        start_tp : int, optional
            The starting time point for the plot, by default 200.
        end_tp : int, optional
            The ending time point for the plot, by default 500.

        Returns
        -------
        None
        """
        X_df_encoded_m_num = df[features].iloc[start_tp:end_tp]
        overall_mean = X_df_encoded_m_num.mean()

        # Calculate rolling mean and standard deviation with a window size
        rolling_mean = X_df_encoded_m_num.rolling(
            window=window_size, min_periods=1
        ).mean()
        rolling_std = X_df_encoded_m_num.rolling(
            window=window_size, min_periods=1
        ).std()

        # Combine the results into a single DataFrame
        rolling_stats = pd.concat([rolling_mean, rolling_std], axis=1)
        rolling_stats.columns = [
            f"{col}_mean" for col in X_df_encoded_m_num.columns
        ] + [f"{col}_std" for col in X_df_encoded_m_num.columns]

        # Predefined list of colors for each subplot
        colors = [
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
            "red",
        ]
        color_cycle = itertools.cycle(colors)

        # Plotting the rolling mean and standard deviation
        n_features = len(features)
        fig, axes = plt.subplots(
            n_features, 1, figsize=(12, 3 * n_features), sharex=True
        )

        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ax = axes[i]
            color = next(color_cycle)

            ax.plot(
                rolling_stats.index,
                rolling_stats[f"{feature}_mean"],
                label=f"{feature} Rolling Mean",
                color=color,
            )
            # Plot standard deviation (scaled for visibility)
            ax.fill_between(
                rolling_stats.index,
                rolling_stats[f"{feature}_mean"] - rolling_stats[f"{feature}_std"],
                rolling_stats[f"{feature}_mean"] + rolling_stats[f"{feature}_std"],
                color=color,
                alpha=0.2,
            )

            # Plot overall mean lines
            ax.hlines(
                overall_mean[feature],
                rolling_stats.index[0],
                rolling_stats.index[-1],
                colors="red",
                linestyles="dotted",
                label=f"{feature} Overall Mean",
            )

            ax.set_title(f"Rolling Mean and Standard Deviation for {feature}")
            ax.set_ylabel("Velocity")
            ax.legend()
            ax.grid(True)

        plt.xlabel("Time Window")
        plt.tight_layout()
        plt.show()


class FeatureVelocity(Velocity):
    def __init__(self, data_obj: CfpdssDataEncoder) -> None:
        """
        Initializes the FeatureVelocity object with the provided data encoder.

        Parameters
        ----------
        data_obj : CfpdssDataEncoder
            The data encoder object containing encoded data and metadata.
        """
        self._data_obj: CfpdssDataEncoder = data_obj

    def plot(self, features: Union[str, Iterable[str]], *args, **kwargs) -> None:
        """
        Plots the velocity of features, either as a stacked bar chart for categorical
        features or rolling means and standard deviations for numerical features.

        Parameters
        ----------
        features : Union[str, Iterable[str]]
            The feature name(s) to plot. If a string is provided, it is treated as a
            categorical feature. If an iterable of strings is provided, it is treated
            as numerical features.
        """
        if isinstance(features, str):
            if features in self._data_obj.original_categorical_cols:
                encoded_feature_name = self._data_obj.categorical_column_mapping[
                    features
                ]
                kwargs["feature"] = encoded_feature_name
                StackedBarChart.plot(df=self._data_obj.X_encoded_data, *args, **kwargs)
                return
            raise ValueError(f"Feature {features} doesn't exist in the given data.")

        # If list/iterable of features is provided, plot rolling means for numerical features
        if isinstance(features, Iterable):
            encoded_features_list = []
            for feature in features:
                if feature not in self._data_obj.original_numerical_cols:
                    if feature in self._data_obj.original_categorical_cols:
                        # Feature is not numerical but categorical
                        raise ValueError(
                            f"Features iterable must contain all numerical features. "
                            f"Feature {feature} is categorical."
                        )
                    # Feature not exists in our dataset
                    raise ValueError(f"Feature {feature} doesn't exists in Data")
                # Feature is numerical
                encoded_features_list.append(
                    self._data_obj.numerical_column_mapping[feature]
                )
            kwargs["features"] = encoded_features_list
            RollingMeansStds.plot(df=self._data_obj.X_encoded_data, *args, **kwargs)
            return

        raise ValueError(
            "Features parameter should be either a string for a categorical feature or "
            "an iterable for numerical features."
        )


class StreamGraph(Velocity):

    def __init__(self, data_obj: CfpdssDataEncoder) -> None:
        """
        Initializes the FeatureVelocity object with the provided data encoder.

        Parameters
        ----------
        data_obj : CfpdssDataEncoder
            The data encoder object containing encoded data and metadata.
        """
        self._data_obj: CfpdssDataEncoder = data_obj

    def get_timepoint(self, window_size):
        time_points = list(range(0, 13000, window_size))
        return time_points

    def count_categories_in_chunks(self, column, chunk_size=50):
        # Initialize an empty dictionary to hold the counts for each category
        category_counts = {}

        # Calculate the number of chunks
        num_chunks = len(column) // chunk_size + (
            1 if len(column) % chunk_size != 0 else 0
        )

        for i in range(num_chunks):
            # Get the start and end indices for the current chunk
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size

            # Get the current chunk
            chunk = column[start_idx:end_idx]

            # Count the occurrences of each category in the current chunk
            chunk_count = chunk.value_counts().to_dict()

            # Update the counts in the category_counts dictionary
            for category, count in chunk_count.items():
                if category not in category_counts:
                    category_counts[category] = [0] * num_chunks
                category_counts[category][i] = count

            # Ensure all categories have a list of the correct length
            for category in category_counts:
                if len(category_counts[category]) < num_chunks:
                    category_counts[category].extend(
                        [0] * (num_chunks - len(category_counts[category]))
                    )

        return category_counts

    def plot(self, feature):
        window_size = 50
        time_points = self.get_timepoint(window_size)
        result = self.count_categories_in_chunks(self._data_obj.X_encoded_data[feature])
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.stackplot(
            time_points,
            result.values(),
            labels=result.keys(),
            alpha=0.8,
            baseline="wiggle",
        )
        ax.legend(loc="upper right", reverse=True)
        ax.set_title(f"Velocity of {feature}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Height")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1000))
        plt.show()


if __name__ == "__main__":
    from stream_viz.data_encoders.cfpdss_data_encoder import (
        CfpdssDataEncoder,
        MissingDataEncoder,
    )
    from stream_viz.utils.constants import _MISSING_DATA_PATH

    # Cfpdss data encoding with missing values
    missing = MissingDataEncoder()
    missing.read_csv_data(filepath_or_buffer=_MISSING_DATA_PATH, index_col=[0])
    missing.encode_data()
    normal = CfpdssDataEncoder()
    normal.read_csv_data(_MISSING_DATA_PATH)
    normal.encode_data()

    # feature_vel = FeatureVelocity(missing)
    # feature_vel.plot(
    #     features="c5", chunk_size=100, start_period=10, end_period=35, x_label_every=5
    # )
    # feature_vel.plot(features=["n0", "n1"], window_size=10, start_tp=200, end_tp=500)

    # stacked_obj = StackedBarChart()
    # cat_feature = "c5_b"
    # stacked_obj.plot_velocity(missing.X_encoded_data, cat_feature, 100, 10, 35)

    # roll_mean_obj = RollingMeansStds()
    # numerical_features = ["n0", "n1"]
    # roll_mean_obj.plot_velocity(missing.X_encoded_data, numerical_features, window_size=10)
    stream_graph = StreamGraph(normal)
    stream_graph.plot("c5_b")
