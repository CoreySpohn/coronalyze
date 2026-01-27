"""Matched-filter SNR estimation (experimental alternative to Mawet).

This module provides an experimental alternative SNR method using Gaussian
matched filtering with annulus-based noise estimation. It is provided for
research comparison but is NOT the standard method.

For production use, prefer the standard Mawet method in coronalyze.core.snr.

Functions:
    - matched_filter_snr(): Calculate SNR using matched-filter approach
    - matched_filter_snr_estimator(): Factory for JIT-ready estimator

Classes:
    - MatchedFilterSNREstimator: Equinox module for efficient batch computation

Note: This module is NOT exported from the main coronalyze namespace.
Import directly: from coronalyze.core.matched_filter import matched_filter_snr
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp

from coronalyze.core.map_coordinates import map_coordinates

# =============================================================================
# Matched Filter SNR Estimator
# =============================================================================


class MatchedFilterSNREstimator(eqx.Module):
    """Matched-filter SNR estimator (experimental).

    Uses Gaussian matched filtering with annulus-based noise estimation.
    This is an alternative to the standard Mawet SNR, useful for:
    - Research comparison with the standard method
    - Exploring different noise models

    Note: For production use, prefer SNREstimator (Mawet method).

    Example:
        >>> from coronalyze.core.matched_filter import matched_filter_snr_estimator
        >>> estimator = matched_filter_snr_estimator(fwhm=4.0)
        >>> snrs = estimator(image, positions)
    """

    # Dynamic fields
    kernel_1d: jnp.ndarray
    fwhm: float

    # Static fields
    order: int = eqx.field(static=True)

    def __init__(
        self,
        fwhm: float,
        fast: bool = False,
    ):
        """Initialize estimator and pre-compute Gaussian kernel.

        Args:
            fwhm: Full width at half maximum in pixels.
            fast: If True, use bilinear interpolation for speed.
        """
        self.fwhm = fwhm
        self.order = 1 if fast else 3

        # Pre-compute Gaussian kernel
        sigma = fwhm / 2.355
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = jnp.arange(kernel_size) - kernel_size // 2
        kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
        self.kernel_1d = kernel_1d / jnp.sum(kernel_1d)

    def __call__(
        self,
        image: jnp.ndarray,
        positions: jnp.ndarray,
        annulus_inner: float | None = None,
        annulus_outer: float | None = None,
    ) -> jnp.ndarray:
        """Calculate matched-filter SNR for a list of positions.

        Args:
            image: 2D science image (should be PSF-subtracted).
            positions: (N, 2) array of (y, x) coordinates.
            annulus_inner: Inner radius of noise annulus (default: auto).
            annulus_outer: Outer radius of noise annulus (default: auto).

        Returns:
            (N,) array of SNR values.
        """
        inner_r = annulus_inner if annulus_inner is not None else -1.0
        outer_r = annulus_outer if annulus_outer is not None else -1.0

        return _matched_filter_snr_batch_core(
            image,
            positions,
            self.fwhm,
            self.kernel_1d,
            self.order,
            inner_r,
            outer_r,
        )


# =============================================================================
# Factory Function
# =============================================================================


def matched_filter_snr_estimator(
    fwhm: float, fast: bool = False
) -> MatchedFilterSNREstimator:
    """Create a matched-filter SNR estimator with pre-computed Gaussian kernel.

    Args:
        fwhm: Full width at half maximum in pixels.
        fast: Use bilinear interpolation for speed.

    Returns:
        MatchedFilterSNREstimator instance.
    """
    return MatchedFilterSNREstimator(fwhm=fwhm, fast=fast)


# =============================================================================
# High-Level Convenience Function
# =============================================================================


def matched_filter_snr(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    fwhm: float,
    annulus_inner: float | None = None,
    annulus_outer: float | None = None,
    fast: bool = False,
) -> jnp.ndarray:
    """Calculate matched-filter SNR at specific positions (experimental).

    Uses Gaussian matched filtering with annulus-based noise estimation.
    This is an experimental alternative to Mawet SNR for research comparison.

    For production use, prefer snr() which uses the standard Mawet method.

    Args:
        image: 2D science image (should be PSF-subtracted).
        positions: (N, 2) array of (y, x) coordinates.
        fwhm: Full width at half maximum in pixels.
        annulus_inner: Inner radius of noise annulus (default: auto).
        annulus_outer: Outer radius of noise annulus (default: auto).
        fast: Use bilinear interpolation for speed.

    Returns:
        (N,) array of SNR values.
    """
    estimator = matched_filter_snr_estimator(fwhm=fwhm, fast=fast)
    return estimator(image, positions, annulus_inner, annulus_outer)


# =============================================================================
# Core JIT-Compiled Functions (Internal)
# =============================================================================


@functools.partial(jax.jit, static_argnums=(4,))
def _matched_filter_snr_batch_core(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    fwhm: float,
    kernel_1d: jnp.ndarray,
    order: int,
    annulus_inner: float,
    annulus_outer: float,
) -> jnp.ndarray:
    """JIT-compiled batch matched-filter SNR calculation."""
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    max_radius = jnp.minimum(ny, nx) / 2 - 1

    # Compute filtered image ONCE
    filtered = _gaussian_filter_2d(image, kernel_1d)

    # Pre-compute coordinate grids
    y_coords, x_coords = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
    r_grid = jnp.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    def _single_snr(planet_pos: jnp.ndarray) -> float:
        py, px = planet_pos[0], planet_pos[1]
        r_planet = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)

        # Default annulus bounds
        default_inner = jnp.maximum(r_planet - fwhm, fwhm)
        default_outer = jnp.minimum(r_planet + fwhm, max_radius)
        default_inner = jnp.minimum(default_inner, default_outer - 1.0)

        inner_r = jnp.where(annulus_inner >= 0, annulus_inner, default_inner)
        outer_r = jnp.where(annulus_outer >= 0, annulus_outer, default_outer)

        # Extract raw signal
        raw_signal = map_coordinates(filtered, jnp.array([[py], [px]]), order=order)[0]

        # Annulus mask
        planet_dist = jnp.sqrt((y_coords - py) ** 2 + (x_coords - px) ** 2)
        annulus_mask = (
            (r_grid >= inner_r) & (r_grid <= outer_r) & (planet_dist > fwhm * 1.5)
        )

        # Background statistics
        masked_vals = jnp.where(annulus_mask, filtered, jnp.nan)
        bg_mean = jnp.nanmean(masked_vals)
        variance = jnp.nanmean(
            jnp.where(annulus_mask, (filtered - bg_mean) ** 2, jnp.nan)
        )
        bg_std = jnp.sqrt(variance)

        bg_mean = jnp.nan_to_num(bg_mean, nan=0.0)
        bg_std = jnp.nan_to_num(bg_std, nan=1.0)

        signal = raw_signal - bg_mean
        return signal / jnp.maximum(bg_std, 1e-10)

    return jax.vmap(_single_snr)(positions)


@jax.jit
def _gaussian_filter_2d(image: jnp.ndarray, kernel_1d: jnp.ndarray) -> jnp.ndarray:
    """Apply 2D Gaussian filter using separable convolution."""
    pad_size = len(kernel_1d) // 2

    # Row convolution
    padded = jnp.pad(image, ((0, 0), (pad_size, pad_size)), mode="reflect")
    row_conv = jax.vmap(lambda row: jnp.convolve(row, kernel_1d, mode="valid"))(padded)

    # Column convolution
    padded = jnp.pad(row_conv, ((pad_size, pad_size), (0, 0)), mode="reflect")
    col_conv = jax.vmap(
        lambda col: jnp.convolve(col, kernel_1d, mode="valid"), in_axes=1, out_axes=1
    )(padded)

    return col_conv
