"""Aperture photometry functions for coronagraphic images."""

import jax.numpy as jnp


def circular_aperture_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> jnp.ndarray:
    """Create a circular aperture mask.

    Args:
        shape: Image shape (ny, nx).
        center: Center of aperture (y, x) in pixels.
        radius: Radius of aperture in pixels.

    Returns:
        Boolean mask array with True inside the aperture.
    """
    ny, nx = shape
    y, x = jnp.ogrid[:ny, :nx]
    cy, cx = center
    distance = jnp.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return distance <= radius


def aperture_photometry(
    image: jnp.ndarray,
    center: tuple[float, float],
    radius: float,
) -> float:
    """Perform circular aperture photometry on an image.

    Args:
        image: 2D image array.
        center: Center of aperture (y, x) in pixels.
        radius: Radius of aperture in pixels.

    Returns:
        Total flux within the aperture.
    """
    mask = circular_aperture_mask(image.shape, center, radius)
    return jnp.sum(image * mask)


def aperture_solid_angle(
    radius_pixels: float,
    pixel_scale_arcsec: float,
) -> float:
    """Calculate the solid angle of a circular aperture.

    Args:
        radius_pixels: Aperture radius in pixels.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.

    Returns:
        Solid angle in arcsec^2.
    """
    radius_arcsec = radius_pixels * pixel_scale_arcsec
    return jnp.pi * radius_arcsec**2
