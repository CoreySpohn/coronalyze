"""SNR calculations for high-contrast coronagraphic imaging.

This module provides SNR estimation using the Mawet et al. (2014) method,
which is the standard approach for exoplanet detection in coronagraphic images.

Primary API:
    - snr(): Calculate SNR at positions
    - snr_map(): Generate 2D SNR detection map
    - snr_estimator(): Factory for JIT-ready SNREstimator objects

Classes:
    - SNREstimator: Equinox module for efficient batch SNR computation

For experimental matched-filter SNR, see coronalyze.core.matched_filter.

Reference: Mawet et al. (2014), ApJ, 792, 97
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp

from coronalyze.core.geometry import generate_aperture_coords, get_center
from coronalyze.core.map_coordinates import map_coordinates
from coronalyze.core.photometry import flux_map, make_aperture_kernel
from coronalyze.core.statistics import masked_mean, masked_std, small_sample_penalty

# =============================================================================
# SNR Estimator Class
# =============================================================================


class SNREstimator(eqx.Module):
    """Standard SNR estimator implementing Mawet et al. (2014).

    Pre-computes the aperture kernel to allow efficient JIT-compilation
    in iterative pipelines. This class is an Equinox module, meaning it
    can be passed into JIT-compiled functions as a PyTree.

    The SNR is calculated using small-sample statistics correction:
        SNR = (x_planet - x_bg_mean) / (sigma_bg * sqrt(1 + 1/n_bg))

    Example::

        # High-performance pipeline usage
        estimator = snr_estimator(fwhm=4.0, fast=True)

        @jax.jit
        def process_cube(images, positions):
            return jax.vmap(lambda img: estimator(img, positions))(images)

        snrs = process_cube(image_cube, planet_positions)

    Reference:
        Mawet et al. (2014), ApJ, 792, 97
    """

    # Dynamic fields (traced by JAX)
    kernel: jnp.ndarray
    fwhm: float
    exclusion_buffer: float

    # Static fields (hashed by JAX, handled by Equinox)
    max_apertures: int = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __init__(
        self,
        fwhm: float,
        soft: bool = True,
        sharpness: float = 10.0,
        fast: bool = False,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
    ):
        """Initialize estimator and pre-compute aperture kernel.

        Args:
            fwhm: Full width at half maximum in pixels (aperture diameter).
            soft: Use differentiable soft aperture edges (default True).
            sharpness: Sigmoid sharpness for soft apertures (default 10.0).
            fast: If True, use bilinear interpolation for ~3x speedup.
                  If False (default), use cubic spline for sub-pixel accuracy.
            max_apertures: Maximum buffer size for static array shapes.
            exclusion_buffer: Gap between test and first reference aperture in
                units of angular step (default 0.5). Prevents PSF wing leakage.
        """
        self.fwhm = fwhm
        self.max_apertures = max_apertures
        self.order = 1 if fast else 3
        self.exclusion_buffer = exclusion_buffer

        # Pre-compute the kernel once (expensive operation)
        self.kernel = make_aperture_kernel(
            radius=fwhm / 2.0,
            soft=soft,
            sharpness=sharpness,
        )

    def __call__(
        self,
        image: jnp.ndarray,
        positions: jnp.ndarray,
        validity_map: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Calculate SNR for a list of candidate positions.

        Args:
            image: 2D science image.
            positions: (N, 2) array of (y, x) coordinates.
            validity_map: Optional 2D mask (1=valid, 0=invalid). Used to exclude
                known companions, bad pixels, or edge regions. Off-chip apertures
                are automatically excluded via boundary handling.

        Returns:
            (N,) array of SNR values.
        """
        return _snr_batch_core(
            image,
            positions,
            self.kernel,
            self.fwhm,
            self.max_apertures,
            self.order,
            self.exclusion_buffer,
            validity_map,
        )

    def map(
        self,
        image: jnp.ndarray,
        validity_map: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Generate a full SNR detection map for the image.

        This is computationally expensive O(N²) but useful for
        generating detection maps.

        Args:
            image: 2D science image.
            validity_map: Optional 2D mask (1=valid, 0=invalid).

        Returns:
            2D array of SNR values matching image shape.
        """
        ny, nx = image.shape
        y_coords, x_coords = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        positions = jnp.stack([y_coords.ravel(), x_coords.ravel()], axis=1)

        flat_snr = self(image, positions, validity_map)
        return flat_snr.reshape(ny, nx)


# =============================================================================
# Factory Function
# =============================================================================


def snr_estimator(
    fwhm: float,
    soft: bool = True,
    sharpness: float = 10.0,
    fast: bool = False,
    max_apertures: int = 200,
    exclusion_buffer: float = 0.5,
) -> SNREstimator:
    """Create a JIT-ready SNR estimator with pre-computed kernel.

    This factory creates an SNREstimator instance that can be efficiently
    used in JIT-compiled pipelines. The aperture kernel is computed once
    at build time.

    Args:
        fwhm: Full width at half maximum in pixels.
        soft: Use soft aperture edges for differentiability.
        sharpness: Sigmoid sharpness for soft apertures.
        fast: Use bilinear interpolation for ~3x speedup.
        max_apertures: Maximum buffer size for static shapes.
        exclusion_buffer: Gap between test and first reference aperture in
            units of angular step (default 0.5). Prevents PSF wing leakage.

    Returns:
        SNREstimator instance ready for use.

    Example::

        estimator = snr_estimator(fwhm=4.0, fast=True)

        @jax.jit
        def pipeline(images, positions):
            return jax.vmap(lambda img: estimator(img, positions))(images)
    """
    return SNREstimator(
        fwhm=fwhm,
        soft=soft,
        sharpness=sharpness,
        fast=fast,
        max_apertures=max_apertures,
        exclusion_buffer=exclusion_buffer,
    )


# =============================================================================
# High-Level Convenience Functions
# =============================================================================


def snr(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    fwhm: float,
    soft: bool = True,
    sharpness: float = 10.0,
    fast: bool = False,
    max_apertures: int = 200,
    exclusion_buffer: float = 0.5,
    validity_map: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate SNR at specific positions using Mawet et al. (2014).

    This is a convenience wrapper around snr_estimator() for simple use cases.
    For iterative pipelines, use snr_estimator() to avoid repeated kernel creation.

    Args:
        image: 2D science image.
        positions: (N, 2) array of (y, x) coordinates.
        fwhm: Full width at half maximum in pixels.
        soft: Use soft aperture edges.
        sharpness: Sigmoid sharpness for soft apertures.
        fast: Use bilinear interpolation for ~3x speedup.
        max_apertures: Maximum buffer size.
        exclusion_buffer: Gap between test and first reference aperture in
            units of angular step (default 0.5). Prevents PSF wing leakage.
        validity_map: Optional 2D mask (1=valid, 0=invalid) to exclude
            known companions, bad pixels, or edge regions.

    Returns:
        (N,) array of SNR values.

    Reference:
        Mawet et al. (2014), ApJ, 792, 97
    """
    estimator = snr_estimator(
        fwhm=fwhm,
        soft=soft,
        sharpness=sharpness,
        fast=fast,
        max_apertures=max_apertures,
        exclusion_buffer=exclusion_buffer,
    )
    return estimator(image, positions, validity_map)


def snr_map(
    image: jnp.ndarray,
    fwhm: float,
    soft: bool = True,
    sharpness: float = 10.0,
    fast: bool = False,
    max_apertures: int = 200,
    exclusion_buffer: float = 0.5,
) -> jnp.ndarray:
    """Generate a 2D map of SNR values using Mawet et al. (2014).

    Computes SNR at every pixel position. This is computationally expensive
    O(N²) but useful for generating detection maps.

    Args:
        image: 2D science image.
        fwhm: Full width at half maximum in pixels.
        soft: Use soft aperture edges.
        sharpness: Sigmoid sharpness.
        fast: Use bilinear interpolation for speedup.
        max_apertures: Maximum buffer size.
        exclusion_buffer: Gap between test and first reference aperture in
            units of angular step (default 0.5). Prevents PSF wing leakage.

    Returns:
        2D array of SNR values, same shape as input image.
    """
    estimator = snr_estimator(
        fwhm=fwhm,
        soft=soft,
        sharpness=sharpness,
        fast=fast,
        max_apertures=max_apertures,
        exclusion_buffer=exclusion_buffer,
    )
    return estimator.map(image)


# =============================================================================
# Core JIT-Compiled Functions (Internal)
# =============================================================================


@functools.partial(jax.jit, static_argnums=(4, 5))
def _snr_batch_core(
    image: jnp.ndarray,
    positions: jnp.ndarray,
    kernel: jnp.ndarray,
    fwhm: float,
    max_apertures: int,
    order: int,
    exclusion_buffer: float = 0.5,
    validity_map: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """JIT-compiled batch SNR calculation (Mawet method).

    Args:
        image: 2D science image.
        positions: (N, 2) array of (y, x) coordinates.
        kernel: Pre-computed aperture kernel.
        fwhm: Full width at half maximum in pixels.
        max_apertures: Maximum buffer size for static shapes.
        order: Interpolation order (1=bilinear, 3=cubic).
        exclusion_buffer: Angular gap between test and first reference.
        validity_map: Optional 2D mask (1=valid, 0=invalid). Off-chip
            locations automatically get 0 via cval boundary handling.
    """
    # Compute flux map ONCE
    fmap = flux_map(image, kernel)
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Default to all-valid if no mask provided
    if validity_map is None:
        validity_map = jnp.ones((ny, nx))

    def _single_snr(planet_pos: jnp.ndarray) -> float:
        py, px = planet_pos[0], planet_pos[1]

        # Extract planet flux
        planet_flux = map_coordinates(fmap, jnp.array([[py], [px]]), order=order)[0]

        # Geometry
        r_pix = jnp.sqrt((py - cy) ** 2 + (px - cx) ** 2)
        planet_angle = jnp.arctan2(py - cy, px - cx)

        # Number of apertures (VIP formula, with exclusion buffer on both sides)
        half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(r_pix, 0.1), 1.0))
        d_theta = 2.0 * half_angle
        n_theoretical = jnp.floor(2 * jnp.pi / jnp.maximum(d_theta, 0.01))
        # Subtract: 1 for planet position + 2*buffer for gap on each side
        n_actual = jnp.maximum(
            (n_theoretical - 1 - 2 * exclusion_buffer).astype(int), 1
        )

        # Generate reference aperture coordinates with exclusion buffer
        idx_grid = jnp.arange(max_apertures)
        angles = planet_angle - (idx_grid + 1 + exclusion_buffer) * d_theta
        ref_y = cy + r_pix * jnp.sin(angles)
        ref_x = cx + r_pix * jnp.cos(angles)

        # Sample validity map (cval=0.0 auto-excludes off-chip apertures)
        ref_validity = map_coordinates(
            validity_map,
            jnp.stack([ref_y, ref_x]),
            order=0,  # Nearest neighbor for speed
            cval=0.0,  # Off-chip = invalid
        )

        # Unified mask: index valid AND spatially valid
        mask = (idx_grid < n_actual) & (ref_validity > 0.5)

        # Sample background fluxes
        ref_fluxes = map_coordinates(fmap, jnp.stack([ref_y, ref_x]), order=order)

        # Masked statistics using actual valid count
        n_valid = jnp.sum(mask)
        bg_mean = jnp.sum(ref_fluxes * mask) / jnp.maximum(n_valid, 1.0)
        residuals = (ref_fluxes - bg_mean) * mask
        bg_std = jnp.sqrt(jnp.sum(residuals**2) / jnp.maximum(n_valid - 1, 1.0))

        # Small-sample penalty using actual valid count
        penalty = jnp.sqrt(1 + 1 / jnp.maximum(n_valid, 1.0))

        # SNR calculation
        signal = planet_flux - bg_mean
        noise = bg_std * penalty
        snr_val = signal / jnp.maximum(noise, 1e-10)

        # Return NaN for unreliable measurements:
        # - Radius smaller than FWHM (can't fit reference apertures)
        # - Fewer than 3 valid reference apertures (insufficient statistics)
        is_valid = (r_pix >= fwhm) & (n_valid >= 3)
        return jnp.where(is_valid, snr_val, jnp.nan)

    return jax.vmap(_single_snr)(positions)


# =============================================================================
# CCD Equation SNR
# =============================================================================


def calculate_ccd_snr(
    signal: float,
    background_noise: float,
    read_noise: float = 0.0,
    dark_current: float = 0.0,
) -> float:
    """Calculate signal-to-noise ratio using the CCD equation.

    Uses the standard CCD equation for SNR:
        SNR = S / sqrt(S + B + R^2 + D)

    where:
        S = signal (electrons)
        B = background noise (electrons)
        R = read noise (electrons)
        D = dark current (electrons)

    This is distinct from Mawet SNR which uses spatial aperture statistics.

    Args:
        signal: Source signal in electrons.
        background_noise: Background noise in electrons (from sky, zodi, etc.).
        read_noise: Read noise in electrons (per pixel, summed over aperture).
        dark_current: Dark current in electrons (summed over aperture).

    Returns:
        Signal-to-noise ratio.
    """
    variance = signal + background_noise + read_noise**2 + dark_current
    return signal / jnp.sqrt(variance)


def exposure_time_for_snr(
    target_snr: float,
    signal_rate: float,
    background_rate: float,
    read_noise: float = 0.0,
    dark_rate: float = 0.0,
) -> float:
    """Calculate exposure time needed to achieve a target SNR.

    Solves the CCD equation for exposure time:
        SNR = S*t / sqrt(S*t + B*t + R^2 + D*t)

    This is a quadratic equation in t.

    Args:
        target_snr: Desired signal-to-noise ratio.
        signal_rate: Source signal rate in electrons/second.
        background_rate: Background rate in electrons/second.
        read_noise: Read noise in electrons (constant, not per second).
        dark_rate: Dark current rate in electrons/second.

    Returns:
        Required exposure time in seconds.
    """
    a = signal_rate**2
    b = -(target_snr**2) * (signal_rate + background_rate + dark_rate)
    c = -(target_snr**2) * read_noise**2

    # Quadratic formula (take positive root)
    discriminant = b**2 - 4 * a * c
    t = (-b + jnp.sqrt(discriminant)) / (2 * a)

    return t
