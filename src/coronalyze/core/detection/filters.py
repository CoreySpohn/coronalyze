"""Concrete detection filters: aperture photometry and Gaussian matched filter."""

import equinox as eqx
import jax
import jax.numpy as jnp
from hwoutils.map_coordinates import map_coordinates

from coronalyze.core.detection.base import AbstractFilter
from coronalyze.core.photometry import flux_map


def gaussian_kernel_1d(fwhm: float) -> jnp.ndarray:
    """Normalized 1D Gaussian kernel for separable matched filtering.

    Identical construction to MatchedFilterSNREstimator (sigma = fwhm/2.355,
    size = next odd integer >= 6 sigma + 1), kept as the single source of
    truth for the Gaussian filter kernel.

    Args:
        fwhm: Full width at half maximum in pixels (concrete Python float).

    Returns:
        1D kernel summing to 1.
    """
    sigma = fwhm / 2.355
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = jnp.arange(kernel_size) - kernel_size // 2
    kernel_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    return kernel_1d / jnp.sum(kernel_1d)


@jax.jit
def gaussian_filter_2d(image: jnp.ndarray, kernel_1d: jnp.ndarray) -> jnp.ndarray:
    """Apply a 2D Gaussian filter using separable convolution.

    Args:
        image: 2D image array.
        kernel_1d: 1D kernel from gaussian_kernel_1d.

    Returns:
        Filtered image, same shape as input.
    """
    pad_size = len(kernel_1d) // 2

    padded = jnp.pad(image, ((0, 0), (pad_size, pad_size)), mode="reflect")
    row_conv = jax.vmap(lambda row: jnp.convolve(row, kernel_1d, mode="valid"))(padded)

    padded = jnp.pad(row_conv, ((pad_size, pad_size), (0, 0)), mode="reflect")
    col_conv = jax.vmap(
        lambda col: jnp.convolve(col, kernel_1d, mode="valid"), in_axes=1, out_axes=1
    )(padded)

    return col_conv


class ApertureFilter(AbstractFilter):
    """Aperture-photometry filter: convolve once, sample the flux map.

    The prepared map is the aperture flux map (convolution with a circular
    kernel); evaluation samples it with sub-pixel interpolation. This is the
    Mawet et al. (2014) signal measurement.
    """

    kernel: jnp.ndarray
    order: int = eqx.field(static=True)

    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """Aperture flux map of the image."""
        return flux_map(image, self.kernel)

    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Sample the flux map at one position."""
        return map_coordinates(filtered, position_yx.reshape(2, 1), order=self.order)[0]


class GaussianFilter(AbstractFilter):
    """Gaussian matched filter: separable convolution plus point sampling."""

    kernel_1d: jnp.ndarray
    order: int = eqx.field(static=True)

    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """Gaussian-filtered image."""
        return gaussian_filter_2d(image, self.kernel_1d)

    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Sample the filtered image at one position."""
        return map_coordinates(filtered, position_yx.reshape(2, 1), order=self.order)[0]
