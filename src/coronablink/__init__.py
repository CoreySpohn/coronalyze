"""coronablink: JAX-based post-processing for coronagraphic direct imaging.

This library provides analysis tools for coronagraphic observations,
designed as a companion to coronagraphoto.

Primary SNR API (Mawet et al. 2014):
    - snr(): Calculate SNR at positions
    - snr_map(): Generate 2D SNR detection map
    - snr_estimator(): Factory for JIT-ready SNREstimator objects

For experimental matched-filter SNR, see coronablink.core.matched_filter.
"""

# Analysis workflows
from coronablink.analysis import (
    get_perfect_residuals,
    get_photon_noise_map,
    simulate_observation,
)

# Core primitives
from coronablink.core import (
    aperture_photometry,
    aperture_solid_angle,
    calculate_n_apertures,
    ccw_rotation_matrix,
    circular_aperture_mask,
    flux_map,
    generate_aperture_coords,
    get_center,
    get_pca_basis,
    inject_planet,
    make_aperture_kernel,
    make_simple_disk,
    masked_mean,
    masked_std,
    pca_subtract,
    radial_distance,
    resample_flux,
    shift_image,
    small_sample_penalty,
    soft_aperture_mask,
)

# Modeling primitives (including subtraction)
from coronablink.core.modeling import (
    subtract_disk,
    subtract_star,
)

# SNR API (Mawet method)
from coronablink.core.snr import (
    SNREstimator,
    calculate_ccd_snr,
    exposure_time_for_snr,
    snr,
    snr_estimator,
    snr_map,
)

# Example data (via pooch)
from coronablink.datasets import fetch_all, fetch_coronagraph, fetch_scene

# Yield pipelines (high-level workflows)
from coronablink.pipelines import (
    calculate_yield_snr,
    klip_subtract,
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
    # Subtraction Primitives
    "subtract_star",
    "subtract_disk",
    "klip_subtract",
    "calculate_yield_snr",
    # Image Transforms
    "ccw_rotation_matrix",
    "resample_flux",
    "shift_image",
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
]
