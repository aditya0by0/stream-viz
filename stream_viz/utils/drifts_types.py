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
    start_tp: int
    end_tp: int


class Drift(TypedDict):
    drift: DriftPeriod


class WarningLvl(TypedDict):
    warning_lvl: DriftPeriod


class LinearDrift(TypedDict):
    linear_drift: DriftPeriod


class SuddenDrift(TypedDict):
    sudden_drift: DriftPeriod


class GradualDrift(TypedDict):
    gradual_drift: DriftPeriod


_rcd_classes = (Drift, WarningLvl)
_fd_classes = (Drift, LinearDrift, SuddenDrift, GradualDrift)
_all_classes = _rcd_classes + _fd_classes

AllDriftType = Union[_all_classes]
FeatureDriftType = Union[_fd_classes]
RealConceptDriftType = Union[_rcd_classes]


def get_rcd_drift_type_keys() -> Set[str]:
    """Get the valid keys for Real Concept Drift types."""
    valid_keys = set()
    for cls in _rcd_classes:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


def get_fd_drift_type_keys() -> Set[str]:
    """Get the valid keys for Feature Drift types."""
    valid_keys = set()
    for cls in _fd_classes:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


def get_all_drift_types_keys() -> Set[str]:
    """Get the valid keys for Feature Drift types."""
    valid_keys = set()
    for cls in _all_classes:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


if __name__ == "__main__":
    # Output: {'linear_drift', 'sudden_drift', 'gradual_drift', 'drift', 'warning_lvl'}
    print(f"All Drift Type keys: {get_all_drift_types_keys()}")
    print(f"Real Concept Drift Type keys: {get_rcd_drift_type_keys()}")
    print(f"Feature Drift Type keys: {get_fd_drift_type_keys()}")
