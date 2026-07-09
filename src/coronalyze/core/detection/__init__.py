"""Composable detection: filters, samplers, significance tests, estimator."""

from coronalyze.core.detection.base import AbstractFilter, AbstractSampler, AbstractTest
from coronalyze.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)

__all__ = [
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "ApertureFilter",
    "GaussianFilter",
    "gaussian_filter_2d",
    "gaussian_kernel_1d",
]
