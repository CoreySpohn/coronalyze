# ruff: noqa: RUF022
"""coronalyze: JAX-based post-processing for coronagraphic direct imaging.

This library provides analysis tools for coronagraphic observations,
designed as a companion to coronagraphoto.

Primary SNR API (Mawet et al. 2014):
    - snr(): Calculate SNR at positions
    - snr_map(): Generate 2D SNR detection map
    - snr_estimator(): Factory for JIT-ready SNREstimator objects

Seam contracts: FrameSet -> AbstractPostProcessing.detect -> DetectionStats.

Matched-filter SNR (matched_filter_snr), the composable detection estimator,
and the PSF-template matched filter (PSFTemplateFilter) are exported at the
top level. YippyTemplateProvider lives in coronalyze.templates.yippy, behind
the optional yippy extra.
"""

from importlib.metadata import version as _get_version

__version__ = _get_version("coronalyze")

# Analysis workflows
from coronalyze.analysis import (
    get_perfect_residuals,
    get_photon_noise_map,
    simulate_observation,
)

# Seam contract types and the post-processing interface
from coronalyze.contracts import DetectionStats, FrameSet

# Core primitives
from coronalyze.core import (
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
    MatchedFilterSNREstimator,
    PSFTemplateFilter,
    TwoSampleTTest,
    aperture_photometry,
    aperture_solid_angle,
    calculate_n_apertures,
    circular_aperture_mask,
    extract_patch,
    flux_map,
    gaussian_filter_2d,
    gaussian_kernel_1d,
    generate_aperture_coords,
    get_center,
    get_pca_basis,
    grubbs_fpf,
    inject_planet,
    make_aperture_kernel,
    make_simple_disk,
    masked_mean,
    masked_std,
    matched_filter_snr,
    matched_filter_snr_estimator,
    n_reference_apertures,
    normal_sf,
    pca_subtract,
    radial_distance,
    small_sample_penalty,
    soft_aperture_mask,
    student_t_sf,
)

# Modeling primitives (including subtraction)
from coronalyze.core.modeling import (
    subtract_disk,
    subtract_star,
)

# SNR API (Mawet method)
from coronalyze.core.snr import (
    SNREstimator,
    calculate_ccd_snr,
    exposure_time_for_snr,
    snr,
    snr_estimator,
    snr_map,
)

# Example data (via pooch)
from coronalyze.datasets import fetch_all, fetch_coronagraph, fetch_scene

# Yield pipelines (high-level workflows)
from coronalyze.pipelines import (
    calculate_yield_snr,
    klip_subtract,
)

# Post-processing seam (interface + arms)
from coronalyze.postproc import (
    AbstractPostProcessing,
    MatchedFilterPostProc,
    MawetPostProcessing,
)

# Post-processing configuration
from coronalyze.pp_config import PPConfig

# Template providers (PSF-template matched filtering)
from coronalyze.templates import AbstractTemplateProvider, ArrayTemplateProvider

__all__ = [
    # SNR Estimator API (Mawet method)
    "snr",
    "snr_map",
    "snr_estimator",
    "SNREstimator",
    # CCD-level SNR
    "calculate_ccd_snr",
    "exposure_time_for_snr",
    # Composable detection (filter / sampler / significance)
    "DetectionEstimator",
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "ApertureFilter",
    "GaussianFilter",
    "PSFTemplateFilter",
    "ApertureSampler",
    "AnnulusSampler",
    "TwoSampleTTest",
    "AnnulusSigmaTest",
    "GrubbsTest",
    "gaussian_kernel_1d",
    "gaussian_filter_2d",
    "extract_patch",
    "matched_filter_snr",
    "matched_filter_snr_estimator",
    "MatchedFilterSNREstimator",
    "normal_sf",
    "grubbs_fpf",
    "n_reference_apertures",
    # Subtraction Primitives
    "subtract_star",
    "subtract_disk",
    "klip_subtract",
    "calculate_yield_snr",
    # PCA/KLIP
    "get_pca_basis",
    "pca_subtract",
    # Forward Modeling
    "inject_planet",
    "make_simple_disk",
    # Yield Analysis
    "get_perfect_residuals",
    "get_photon_noise_map",
    "simulate_observation",
    # Example data
    "fetch_all",
    "fetch_coronagraph",
    "fetch_scene",
    # Seam contracts (post-processing interface)
    "FrameSet",
    "DetectionStats",
    "AbstractPostProcessing",
    "MawetPostProcessing",
    "MatchedFilterPostProc",
    "student_t_sf",
    # Template providers (PSF-template matched filtering)
    "AbstractTemplateProvider",
    "ArrayTemplateProvider",
    # Core utilities
    "calculate_n_apertures",
    "flux_map",
    "generate_aperture_coords",
    "get_center",
    "make_aperture_kernel",
    "masked_mean",
    "masked_std",
    "radial_distance",
    "small_sample_penalty",
    # Photometry
    "aperture_photometry",
    "aperture_solid_angle",
    "circular_aperture_mask",
    "soft_aperture_mask",
    # Post-processing configuration
    "PPConfig",
]
