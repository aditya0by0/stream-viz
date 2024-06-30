from typing import Set, TypedDict, Union

__all__ = [
    "DriftPeriod",
    "LinearDrift",
    "SuddenDrift",
    "GradualDrift",
    "DriftType",
    "get_valid_keys",
]


class DriftPeriod(TypedDict):
    start_tp: int
    end_tp: int


class Drift(TypedDict):
    drift: DriftPeriod


class LinearDrift(TypedDict):
    linear_drift: DriftPeriod


class SuddenDrift(TypedDict):
    sudden_drift: DriftPeriod


class GradualDrift(TypedDict):
    gradual_drift: DriftPeriod


DriftType = Union[LinearDrift, SuddenDrift, GradualDrift, Drift]


def get_valid_keys() -> Set[str]:
    """Get the valid keys for DriftType"""
    valid_keys = set()
    for cls in [LinearDrift, SuddenDrift, GradualDrift, Drift]:
        valid_keys.update(cls.__annotations__.keys())
    return valid_keys


if __name__ == "__main__":
    # Output: {'linear_drift', 'sudden_drift', 'gradual_drift', 'drift'}
    print(get_valid_keys())
