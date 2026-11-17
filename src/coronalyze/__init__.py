# ruff: noqa: RUF022
"""coronalyze: JAX-based post-processing for coronagraphic direct imaging.

This library provides analysis tools for coronagraphic observations,
designed as a companion to coronagraphoto.

DetectionEstimator.mawet and DetectionEstimator.matched_filter build
JIT-ready estimators for aperture photometry and Gaussian matched filtering.
Call an estimator at candidate positions or use its map method for a 2D
statistic image.

Seam contracts: FrameSet -> AbstractPostProcessing.detect -> DetectionStats.

The PSF-template matched filter (PSFTemplateFilter) is exported at the top
level. YippyTemplateProvider lives in coronalyze.templates.yippy, behind
the optional yippy extra. The experimental calibration tools (candidate
finding, null counting, injection throughput) live in coronalyze.calibration.
"""

from importlib.metadata import version as _get_version

__version__ = _get_version("coronalyze")

# Analysis workflows
from coronalyze.analysis import (
    get_perfect_residuals,
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
    MapFilter,
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
    get_pca_basis,
    grubbs_fpf,
    inject_planet,
    make_aperture_kernel,
    make_simple_disk,
    masked_mean,
    masked_std,
    n_reference_apertures,
    normal_sf,
    pca_subtract,
    small_sample_penalty,
    soft_aperture_mask,
    student_t_sf,
)

# Modeling primitives (including subtraction)
from coronalyze.core.modeling import (
    subtract_disk,
    subtract_star,
)

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

# Template providers (PSF-template matched filtering)
from coronalyze.templates import AbstractTemplateProvider, ArrayTemplateProvider

__all__ = [
    # Composable detection (filter / sampler / significance)
    "DetectionEstimator",
    "AbstractFilter",
    "AbstractSampler",
    "AbstractTest",
    "ApertureFilter",
    "GaussianFilter",
    "MapFilter",
    "PSFTemplateFilter",
    "ApertureSampler",
    "AnnulusSampler",
    "TwoSampleTTest",
    "AnnulusSigmaTest",
    "GrubbsTest",
    "gaussian_kernel_1d",
    "gaussian_filter_2d",
    "extract_patch",
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
    "make_aperture_kernel",
    "masked_mean",
    "masked_std",
    "small_sample_penalty",
    # Photometry
    "aperture_photometry",
    "aperture_solid_angle",
    "circular_aperture_mask",
    "soft_aperture_mask",
]
