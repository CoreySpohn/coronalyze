"""Geometric utilities for coronagraphic image analysis.

Functions for coordinate transforms, distance calculations, and generating
aperture positions with static array shapes for JAX compatibility.
"""

import jax.numpy as jnp


def get_center(shape: tuple[int, int]) -> tuple[float, float]:
    """Get the center coordinates of an image (0-indexed geometric center).

    For an image of size N, the geometric center is at (N-1)/2.
    This matches the convention used in modeling.py for planet injection.

    Args:
        shape: Image shape (ny, nx).

    Returns:
        Center coordinates (cy, cx).
    """
    return ((shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0)


def radial_distance(
    shape: tuple[int, int],
    center: tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Calculate radial distance from center for each pixel.

    Args:
        shape: Image shape (ny, nx).
        center: Center coordinates (cy, cx). If None, uses image center.

    Returns:
        2D array of radial distances in pixels.
    """
    ny, nx = shape
    if center is None:
        center = get_center(shape)
    cy, cx = center

    y, x = jnp.ogrid[:ny, :nx]
    return jnp.sqrt((y - cy) ** 2 + (x - cx) ** 2)


def calculate_n_apertures(
    radius: float,
    fwhm: float,
    exclusion_buffer: float = 0.5,
) -> int:
    """Calculate the number of reference apertures at a given radius.

    Uses the Mawet et al. (2014) formula with exclusion buffer correction
    to ensure no overlap with the planet aperture on either side.

    This function provides the canonical calculation used by the SNR module,
    ensuring consistency between visualization and computation.

    Args:
        radius: Radial distance from center in pixels.
        fwhm: Full width at half maximum in pixels.
        exclusion_buffer: Gap between planet and first/last reference aperture
            in units of angular step (default 0.5). Creates a gap on both sides.

    Returns:
        Number of valid reference apertures.

    Example:
        >>> from coronablink.core.geometry import calculate_n_apertures
        >>> n = calculate_n_apertures(radius=20, fwhm=5.0)
        >>> print(f"{n} reference apertures at r=20px")
    """
    import numpy as np

    half_angle = np.arcsin(min(fwhm / 2.0 / max(radius, 0.1), 1.0))
    d_theta = 2.0 * half_angle
    n_theoretical = np.floor(2 * np.pi / max(d_theta, 0.01))
    # Subtract: 1 for planet position + 2*buffer for gap on each side
    n_actual = max(int(n_theoretical - 1 - 2 * exclusion_buffer), 1)
    return n_actual


def generate_aperture_coords(
    center: tuple[float, float],
    radius: float,
    planet_angle: float,
    n_apertures: int,
    max_apertures: int = 200,
    fwhm: float | None = None,
    exclusion_buffer: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate coordinates for reference apertures at a given radius.

    Uses a fixed-size array with masking for JAX compatibility (static shapes).
    Apertures are distributed evenly around the annulus, excluding the planet position.

    Matches VIP's clockwise rotation and angle formula from planet position.

    Args:
        center: Image center (cy, cx) in pixels.
        radius: Radial distance from center in pixels.
        planet_angle: Angle of the planet position in radians.
        n_apertures: Actual number of valid apertures to use.
        max_apertures: Maximum buffer size for static array shape.
        fwhm: Full width half maximum for VIP-style angle calculation. If None,
              uses uniform distribution.
        exclusion_buffer: Gap between test and first reference aperture in
            units of angular step (default 0.0). Prevents PSF wing leakage.

    Returns:
        Tuple of (y_coords, x_coords, mask) where:
            - y_coords: Y coordinates of aperture centers (size max_apertures)
            - x_coords: X coordinates of aperture centers (size max_apertures)
            - mask: Boolean mask indicating valid apertures
    """
    cy, cx = center
    idx_grid = jnp.arange(max_apertures)

    # Angular step calculation
    # VIP formula: angle = 2 * arcsin(fwhm/2/radius)
    if fwhm is not None:
        half_angle = jnp.arcsin(jnp.minimum(fwhm / 2.0 / jnp.maximum(radius, 0.1), 1.0))
        d_theta = 2.0 * half_angle
    else:
        # Fallback: uniform distribution
        d_theta = 2 * jnp.pi / jnp.maximum(n_apertures + 1, 1)

    # Angles starting from one step past the planet (with optional buffer)
    # Use NEGATIVE rotation (clockwise) to match VIP's convention
    angles = planet_angle - (idx_grid + 1 + exclusion_buffer) * d_theta

    # Compute coordinates
    y_coords = cy + radius * jnp.sin(angles)
    x_coords = cx + radius * jnp.cos(angles)

    # Mask: valid if index < n_apertures
    mask = idx_grid < n_apertures

    return y_coords, x_coords, mask
