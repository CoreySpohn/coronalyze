"""Core JAX-based analysis primitives for coronalyze.

This module contains pure JAX mathematical functions with no external dependencies.
All functions are JIT-compilable and differentiable.
"""

from coronalyze.core.geometry import (
    calculate_n_apertures,
    generate_aperture_coords,
    get_center,
    radial_distance,
)
from coronalyze.core.image_transforms import (
    ccw_rotation_matrix,
    resample_flux,
    shift_image,
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
from coronalyze.core.statistics import masked_mean, masked_std, small_sample_penalty

__all__ = [
    # SNR Estimator API (Mawet method)
    "snr",
    "snr_map",
    "snr_estimator",
    "SNREstimator",
    # CCD-level SNR
    "calculate_ccd_snr",
    "exposure_time_for_snr",
    # Geometry
    "calculate_n_apertures",
    "generate_aperture_coords",
    "get_center",
    "radial_distance",
    # Image Transforms
    "ccw_rotation_matrix",
    "resample_flux",
    "shift_image",
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
    "masked_mean",
    "masked_std",
    "small_sample_penalty",
]
