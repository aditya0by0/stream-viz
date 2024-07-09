from typing import Set, TypedDict, Union

__all__ = [
    "DriftPeriod",
    "WarningLvl",
    "LinearDrift",
    "SuddenDrift",
    "GradualDrift",
    "AllDriftType",
    "FeatureDriftType",
    "RealConceptDriftType",
    "get_rcd_drift_type_keys",
    "get_fd_drift_type_keys",
    "get_all_drift_types_keys",
]


class DriftPeriod(TypedDict):
    """
    Represents the start and end time points of a drift period.

    Attributes:
    ----------
    start_tp : int
        Start time point of the drift period.
    end_tp : int
        End time point of the drift period.
    """

    start_tp: int
    end_tp: int


class Drift(TypedDict):
    """
    Represents a generic drift type with a drift period.

    Attributes:
    ----------
    drift : DriftPeriod
        Drift period defined by start and end time points.
    """

    drift: DriftPeriod


class WarningLvl(TypedDict):
    """
    Represents a warning level associated with a drift period.

    Attributes:
    ----------
    warning_lvl : DriftPeriod
        Warning level period defined by start and end time points.
    """

    warning_lvl: DriftPeriod


class LinearDrift(TypedDict):
    """
    Represents a linear drift type with a drift period.

    Attributes:
    ----------
    linear_drift : DriftPeriod
        Linear drift period defined by start and end time points.
    """

    linear_drift: DriftPeriod


class SuddenDrift(TypedDict):
    """
    Represents a sudden drift type with a drift period.

    Attributes:
    ----------
    sudden_drift : DriftPeriod
        Sudden drift period defined by start and end time points.
    """

    sudden_drift: DriftPeriod


class GradualDrift(TypedDict):
    """
    Represents a gradual drift type with a drift period.

    Attributes:
    ----------
    gradual_drift : DriftPeriod
        Gradual drift period defined by start and end time points.
    """

    gradual_drift: DriftPeriod


_rcd_classes = (Drift, WarningLvl)
_fd_classes = (Drift, LinearDrift, SuddenDrift, GradualDrift)
_all_classes = _rcd_classes + _fd_classes

AllDriftType = Union[_all_classes]
FeatureDriftType = Union[_fd_classes]
RealConceptDriftType = Union[_rcd_classes]


def get_rcd_drift_type_keys() -> Set[str]:
    """
    Get the valid keys for Real Concept Drift types.

    Returns:
    -------
    Set[str]
        A set of valid keys present in Real Concept Drift type definitions.
    """
    valid_keys = set()
    for cls in _rcd_classes:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


def get_fd_drift_type_keys() -> Set[str]:
    """
    Get the valid keys for Feature Drift types.

    Returns:
    -------
    Set[str]
        A set of valid keys present in Feature Drift type definitions.
    """
    valid_keys = set()
    for cls in _fd_classes:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


def get_all_drift_types_keys() -> Set[str]:
    """
    Get the valid keys for all Drift types.

    Returns:
    -------
    Set[str]
        A set of valid keys present in all Drift type definitions.
    """
    valid_keys = set()
    for cls in _all_classes:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


if __name__ == "__main__":
    # Output: {'linear_drift', 'sudden_drift', 'gradual_drift', 'drift', 'warning_lvl'}
    print(f"All Drift Type keys: {get_all_drift_types_keys()}")
    print(f"Real Concept Drift Type keys: {get_rcd_drift_type_keys()}")
    print(f"Feature Drift Type keys: {get_fd_drift_type_keys()}")
