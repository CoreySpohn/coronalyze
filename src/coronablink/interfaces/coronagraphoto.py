"""Interface for coronagraphoto simulation outputs.

Provides adapters to extract analysis-ready data from coronagraphoto
Exposure and simulation outputs.
"""

import jax.numpy as jnp

from coronablink.core.snr import snr


def extract_image(detector_readout: jnp.ndarray) -> jnp.ndarray:
    """Convert detector readout to an analysis-ready image.

    Currently a pass-through, but can be extended to handle:
    - Multiple detector planes
    - Unit conversions
    - Bad pixel masking

    Args:
        detector_readout: Raw detector output from coronagraphoto simulation.

    Returns:
        2D image array suitable for analysis.
    """
    # Ensure 2D
    if detector_readout.ndim > 2:
        # Take the first plane if datacube
        return detector_readout[0]
    return detector_readout


def get_fwhm(
    wavelength_nm: float,
    diameter_m: float = 6.0,
    pixel_scale_mas: float = 21.8,
) -> float:
    """Calculate the FWHM of the PSF in pixels.

    Uses the diffraction limit: FWHM ≈ λ/D

    Args:
        wavelength_nm: Observation wavelength in nanometers.
        diameter_m: Primary aperture diameter in meters.
        pixel_scale_mas: Pixel scale in milliarcseconds per pixel.

    Returns:
        FWHM in pixels.
    """
    # λ/D in radians
    wavelength_m = wavelength_nm * 1e-9
    lambda_over_d_rad = wavelength_m / diameter_m

    # Convert to milliarcseconds
    lambda_over_d_mas = lambda_over_d_rad * (180 / jnp.pi) * 3600 * 1000

    # FWHM in pixels
    fwhm_pixels = lambda_over_d_mas / pixel_scale_mas

    return fwhm_pixels


def analyze_observation(
    image: jnp.ndarray,
    planet_pos: tuple[float, float],
    wavelength_nm: float,
    diameter_m: float = 6.0,
    pixel_scale_mas: float = 21.8,
    max_apertures: int = 200,
) -> float:
    """High-level analysis: compute Mawet SNR for a planet detection.

    Convenience wrapper that combines FWHM calculation with SNR computation.

    Args:
        image: 2D detector image.
        planet_pos: Planet position (y, x) in pixels.
        wavelength_nm: Observation wavelength in nanometers.
        diameter_m: Primary aperture diameter in meters.
        pixel_scale_mas: Pixel scale in milliarcseconds per pixel.
        max_apertures: Maximum buffer size for static array shapes.

    Returns:
        Signal-to-noise ratio using Mawet 2014 methodology.
    """
    fwhm = get_fwhm(wavelength_nm, diameter_m, pixel_scale_mas)
    # Convert single position to batch format
    positions = jnp.array([[planet_pos[0], planet_pos[1]]])
    return snr(image, positions, fwhm, max_apertures=max_apertures)[0]
