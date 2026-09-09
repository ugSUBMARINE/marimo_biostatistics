"""Reusable statistical calculations for the companion notebooks."""

from typing import Any

import numpy as np


def _validated_sample(values: Any, name: str) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if sample.size < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if not np.isfinite(sample).all():
        raise ValueError(f"{name} must contain only finite values")
    return sample


def pooled_sample_standard_deviation(group_1: Any, group_2: Any) -> float:
    """Return the equal-variance pooled sample standard deviation.

    The two group variances use Bessel's correction.  This quantity is suitable
    for the conventional independent-groups definition of sample Cohen's d.
    """

    sample_1 = _validated_sample(group_1, "group_1")
    sample_2 = _validated_sample(group_2, "group_2")
    degrees_of_freedom = sample_1.size + sample_2.size - 2
    pooled_variance = (
        (sample_1.size - 1) * np.var(sample_1, ddof=1)
        + (sample_2.size - 1) * np.var(sample_2, ddof=1)
    ) / degrees_of_freedom
    if pooled_variance <= 0:
        raise ValueError("pooled variance must be positive")
    return float(np.sqrt(pooled_variance))


def cohen_d_sample(group_1: Any, group_2: Any) -> float:
    """Return sample Cohen's d as ``(mean(group_2) - mean(group_1)) / pooled_sd``."""

    sample_1 = _validated_sample(group_1, "group_1")
    sample_2 = _validated_sample(group_2, "group_2")
    pooled_sd = pooled_sample_standard_deviation(sample_1, sample_2)
    return float((np.mean(sample_2) - np.mean(sample_1)) / pooled_sd)


__all__ = ["cohen_d_sample", "pooled_sample_standard_deviation"]
