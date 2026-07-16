# ruff: noqa: RUF022
"""Core JAX-based analysis primitives for coronalyze.

This module contains pure JAX mathematical functions built on jax, equinox,
and hwoutils only. All functions are JIT-compilable and differentiable.
"""

# Composable detection
from coronalyze.core.detection import (
    AbstractFilter,
    AbstractSampler,
    AbstractTest,
    AnnulusSampler,
    AnnulusSigmaTest,
    ApertureFilter,
    ApertureSampler,
    DetectionEstimator,
    GaussianFilter,
    GrubbsTest,
    TwoSampleTTest,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)
from coronalyze.core.geometry import (
    calculate_n_apertures,
    generate_aperture_coords,
    get_center,
    n_reference_apertures,
    radial_distance,
)

# Matched filter (Gaussian)
from coronalyze.core.matched_filter import (
    MatchedFilterSNREstimator,
    matched_filter_snr,
    matched_filter_snr_estimator,
)
from coronalyze.core.modeling import (
    inject_planet,
    make_simple_disk,
    subtract_disk,
    subtract_star,
)
from coronalyze.core.pca import get_pca_basis, pca_subtract
from coronalyze.core.photometry import (
    aperture_photometry,
    aperture_solid_angle,
    circular_aperture_mask,
    flux_map,
    make_aperture_kernel,
    soft_aperture_mask,
)

# SNR API (Mawet method only)
from coronalyze.core.snr import (
    SNREstimator,
    calculate_ccd_snr,
    exposure_time_for_snr,
    snr,
    snr_estimator,
    snr_map,
)
from coronalyze.core.statistics import (
    grubbs_fpf,
    masked_mean,
    masked_std,
    nanmasked_mean,
    nanmasked_population_std,
    normal_sf,
    small_sample_penalty,
    student_t_sf,
)

__all__ = [
    # SNR Estimator API (Mawet method)
    "snr",
    "snr_map",
    "snr_estimator",
    "SNREstimator",
    # CCD-level SNR
    "calculate_ccd_snr",
    "exposure_time_for_snr",
    # Composable detection
    "DetectionEstimator",
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "ApertureFilter",
    "GaussianFilter",
    "ApertureSampler",
    "AnnulusSampler",
    "TwoSampleTTest",
    "AnnulusSigmaTest",
    "GrubbsTest",
    "gaussian_kernel_1d",
    "gaussian_filter_2d",
    # Matched filter (Gaussian)
    "matched_filter_snr",
    "matched_filter_snr_estimator",
    "MatchedFilterSNREstimator",
    # Geometry
    "calculate_n_apertures",
    "generate_aperture_coords",
    "get_center",
    "n_reference_apertures",
    "radial_distance",
    # PCA
    "get_pca_basis",
    "pca_subtract",
    # Modeling
    "inject_planet",
    "make_simple_disk",
    "subtract_star",
    "subtract_disk",
    # Photometry
    "aperture_photometry",
    "aperture_solid_angle",
    "circular_aperture_mask",
    "flux_map",
    "make_aperture_kernel",
    "soft_aperture_mask",
    # Statistics
    "grubbs_fpf",
    "masked_mean",
    "masked_std",
    "nanmasked_mean",
    "nanmasked_population_std",
    "normal_sf",
    "small_sample_penalty",
    "student_t_sf",
]
