"""Composable detection: filters, samplers, significance tests, estimator."""

from coronalyze.core.detection.base import AbstractFilter, AbstractSampler, AbstractTest
from coronalyze.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)
from coronalyze.core.detection.samplers import AnnulusSampler, ApertureSampler

__all__ = [
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "AnnulusSampler",
    "ApertureFilter",
    "ApertureSampler",
    "GaussianFilter",
    "gaussian_filter_2d",
    "gaussian_kernel_1d",
]
