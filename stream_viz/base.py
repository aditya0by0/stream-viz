from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

# This library enables interactive plots in the notebook.
# Enables functionality to hover over data points to see values, zoom in/out, and pan the plot.
import mpld3
import pandas as pd
from IPython.core.display_functions import display
from ipywidgets import HBox, IntSlider, SelectMultiple, VBox, interactive_output

from stream_viz.utils.drifts_types import AllDriftType, get_all_drift_types_keys

# mpld3.enable_notebook() # Disabled as of now, due to issues with other plots
mpld3.disable_notebook()


class Plotter(ABC):
    """
    Abstract class for plotting data.

    This class defines the interface for plotting data. Subclasses must implement the `plot` method.
    """

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        """
        Abstract method to plot data.
        """
        pass


@dataclass
class DataEncoder(ABC):
    """
    Abstract class for encoding data.

    This class handles reading and encoding data. Subclasses must implement the `encode_data` method.
    """

    def __init__(self):
        self._original_data: pd.DataFrame = pd.DataFrame()
        self._encoded_data: pd.DataFrame = pd.DataFrame()

    def read_csv_data(self, *args, **kwargs) -> None:
        """
        Reads CSV data and stores it in the original_data attribute.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to pd.read_csv.
        **kwargs : dict
            Keyword arguments to pass to pd.read_csv.

        Raises
        ------
        ValueError
            If the data is None or empty.
        """
        data = pd.read_csv(*args, **kwargs)
        if data is None or data.empty:
            raise ValueError("Unable to read data")
        self.original_data = data

    @abstractmethod
    def encode_data(self, *args, **kwargs) -> None:
        """
        Abstract method to encode data.
        """
        pass

    @property
    def original_data(self) -> pd.DataFrame:
        """
        Returns the original data.

        Returns
        -------
        pd.DataFrame
            The original data.

        Raises
        ------
        ValueError
            If the original data is empty.
        """
        if self._original_data is None or self._original_data.empty:
            raise ValueError(
                "Original Data is empty. Please call `read_csv_data` method"
            )
        return self._original_data

    @original_data.setter
    def original_data(self, value: pd.DataFrame) -> None:
        """
        Sets the original data.

        Parameters
        ----------
        value : pd.DataFrame
            The data to set as the original data.

        Raises
        ------
        ValueError
            If the value is not a pandas DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("original_data must be a pandas DataFrame")
        self._original_data = value

    @property
    def encoded_data(self) -> pd.DataFrame:
        """
        Returns the encoded data.

        Returns
        -------
        pd.DataFrame
            The encoded data.

        Raises
        ------
        ValueError
            If the encoded data is empty.
        """
        if self._encoded_data is None or self._encoded_data.empty:
            raise ValueError("Encoded Data is empty. Please call `encode_data` method")
        return self._encoded_data

    @encoded_data.setter
    def encoded_data(self, value: pd.DataFrame) -> None:
        """
        Sets the encoded data.

        Parameters
        ----------
        value : pd.DataFrame
            The data to set as the encoded data.

        Raises
        ------
        ValueError
            If the value is not a pandas DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("encoded_data must be a pandas DataFrame")
        self._encoded_data = value


class DriftDetector(Plotter, ABC):
    """
    Abstract class for detecting and plotting data drift.

    This class defines the interface for drift detection and plotting. Subclasses must implement
    the `update`, `detect_drift`, `plot`, and `drift_records` methods.
    """

    def __init__(self):
        super().__init__()
        self._drift_records: List[AllDriftType] = []
        self._valid_keys: set[str] = get_all_drift_types_keys()

    @abstractmethod
    def update(self, x_i: Dict, y_i: int, tpt: int) -> None:
        """
        Abstract method to update the detector with new data.
        """
        pass

    @abstractmethod
    def detect_drift(self, tpt: int, *args, **kwargs) -> None:
        """
        Abstract method to detect drift.
        """
        pass

    @abstractmethod
    def plot(self, start_tpt: int, end_tpt: int, *args, **kwargs) -> None:
        """
        Abstract method to plot drift data.
        """
        pass

    @property
    @abstractmethod
    def drift_records(self) -> List[AllDriftType]:
        """
        Getter for drift records.

        Returns
        -------
        List[AllDriftType]
            The list of drift records.
        """
        pass

    @drift_records.setter
    @abstractmethod
    def drift_records(self, drift_record: AllDriftType) -> None:
        """
        Setter for drift records. Adds a new drift to the list.

        Parameters
        ----------
        drift_record : AllDriftType
            The drift record to add.
        """
        pass

    def _validate_drift(self, drift: AllDriftType) -> bool:
        """
        Validates the drift record.

        Parameters
        ----------
        drift : AllDriftType
            The drift record to validate.

        Returns
        -------
        bool
            True if the drift record is valid, False otherwise.
        """
        if not any(key in drift for key in self._valid_keys):
            return False
        for key in self._valid_keys:
            if key in drift:
                if "start_tp" in drift[key] and "end_tp" in drift[key]:
                    return True
        return False


class Streamer(ABC):
    """
    Abstract class for streaming data.

    This class defines the interface for streaming data. Subclasses must implement the `stream_data` method.
    """

    def __init__(
        self,
        rcd_detector_obj: Optional[DriftDetector] = None,
        fd_detector_obj: Optional[DriftDetector] = None,
    ):
        self.rcd_detector_obj: Optional[DriftDetector] = rcd_detector_obj
        self.fd_detector_obj: Optional[DriftDetector] = fd_detector_obj

    @abstractmethod
    def stream_data(self, X_df: pd.DataFrame, y_df: pd.Series) -> None:
        """
        Abstract method to stream data.

        Parameters
        ----------
        X_df : pd.DataFrame
            The feature data to stream.
        y_df : pd.Series
            The target data to stream.
        """
        pass


class Velocity(Plotter, ABC):
    """
    Abstract class for plotting velocity data.

    This class defines the interface for plotting velocity data.
    """

    pass


class StrategyPlot(Plotter, ABC):
    """
    Abstract class for plotting strategy data.

    This class defines the interface for plotting strategy data.
    """

    pass


class Binning(ABC):
    """
    Abstract class for performing data binning.

    This class defines the interface for data binning. Subclasses must implement the `perform_binning` method.
    """

    def __init__(self, **kwargs):
        self._bin_thresholds: List[float] = []
        self._binned_data_X: pd.DataFrame = pd.DataFrame()
        # regex pattern for the columns name that needs to be binned
        self._col_name_regex: str = kwargs.get("col_name_regex", r"^n")
        # New name for the columns to be binned
        self._bin_col_names: str = kwargs.get("bin_col_names", r"bin_idx_")

    @abstractmethod
    def perform_binning(self, *args, **kwargs) -> None:
        """
        Abstract method to perform binning on data.
        """
        pass

    @property
    def bin_thresholds(self) -> List[float]:
        """
        Returns the bin thresholds.

        Returns
        -------
        List[float]
            The list of bin thresholds.

        Raises
        ------
        ValueError
            If the bin thresholds are empty.
        """
        if self._bin_thresholds is None:
            raise ValueError("bin_thresholds is empty")
        return self._bin_thresholds

    @bin_thresholds.setter
    def bin_thresholds(self, value: List[float]) -> None:
        """
        Sets the bin thresholds.

        Parameters
        ----------
        value : List[float]
            The list of bin thresholds.

        Raises
        ------
        ValueError
            If the value is not a list.
        """
        if not isinstance(value, list):
            raise ValueError("bin_thresholds must be a List")
        self._bin_thresholds = value

    @property
    def binned_data_X(self) -> pd.DataFrame:
        """
        Returns the binned data.

        Returns
        -------
        pd.DataFrame
            The binned data.

        Raises
        ------
        ValueError
            If the binned data is empty.
        """
        if self._binned_data_X is None or self._binned_data_X.empty:
            raise ValueError("Binned Data is empty.")
        return self._binned_data_X

    @binned_data_X.setter
    def binned_data_X(self, value: pd.DataFrame) -> None:
        """
        Sets the binned data.

        Parameters
        ----------
        value : pd.DataFrame
            The data to set as the binned data.

        Raises
        ------
        ValueError
            If the value is not a pandas DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        self._binned_data_X = value


class InteractivePlot(Plotter, ABC):
    """
    Abstract class for creating interactive plots.

    This class defines the interface for creating interactive plots using widgets. Subclasses must implement
    the `_add_interactive_plot` method.
    """

    def __init__(self, data_df: pd.DataFrame):
        self._data_df: pd.DataFrame = data_df
        self._default_variables: int = 3
        self._add_sliders()
        self._add_feature_selector()

    @abstractmethod
    def _add_interactive_plot(self, *args, **kwargs) -> None:
        """
        Abstract method to add the interactive plot.

        This method links the widgets to the plotting function.
        """
        self.interactive_plot = interactive_output(
            self.plot,
            {
                "start": self.start_slider,
                "end": self.end_slider,
                "features": self.feature_selector,
                **kwargs,
            },
        )

    def _add_sliders(self) -> None:
        """
        Adds sliders for start and end timepoints.
        """
        self.start_slider = IntSlider(
            min=0,
            max=self._data_df.shape[0] - 1,
            step=1,
            value=0,
            description="Start",
        )
        self.end_slider = IntSlider(
            min=0,
            max=self._data_df.shape[0] - 1,
            step=1,
            value=1000,
            description="End",
        )
        # Ensure the end slider always has a value greater than the start slider
        self.start_slider.observe(self._update_end_range, "value")

    def _add_feature_selector(self) -> None:
        """
        Adds a feature selector to select features dynamically.
        """
        self.feature_selector = SelectMultiple(
            options=self._data_df.columns,
            value=tuple(self._data_df.columns[: self._default_variables]),
            description="Features",
            style={"description_width": "initial"},
        )

    def _update_end_range(self, *args) -> None:
        """
        Updates the end range of the slider.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        """
        self.end_slider.min = self.start_slider.value + 1

    def display(self, *args, **kwargs) -> None:
        """
        Displays the interactive plot.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        self._add_interactive_plot(*args, **kwargs)
        widgets_box = HBox(
            [VBox([self.start_slider, self.end_slider]), self.feature_selector]
        )
        display(widgets_box, self.interactive_plot)
