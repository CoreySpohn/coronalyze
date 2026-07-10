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

import equinox as eqx
import jax.numpy as jnp

from coronalyze.core.detection.estimator import DetectionEstimator
from coronalyze.core.detection.filters import GaussianFilter
from coronalyze.core.detection.samplers import AnnulusSampler
from coronalyze.core.detection.significance import AnnulusSigmaTest

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

    Example::

        from coronalyze.core.matched_filter import matched_filter_snr_estimator
        estimator = matched_filter_snr_estimator(fwhm=4.0)
        snrs = estimator(image, positions)
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
        composed = DetectionEstimator(
            filter=GaussianFilter(kernel_1d=self.kernel_1d, order=self.order),
            sampler=AnnulusSampler(
                fwhm=self.fwhm, annulus_inner=inner_r, annulus_outer=outer_r
            ),
            test=AnnulusSigmaTest(),
        )
        return composed(image, positions).statistic


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
