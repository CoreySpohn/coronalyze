"""Tests for detection filter primitives."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.map_coordinates import map_coordinates

from coronalyze.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_filter_2d,
    gaussian_kernel_1d,
)
from coronalyze.core.matched_filter import MatchedFilterSNREstimator
from coronalyze.core.photometry import flux_map, make_aperture_kernel


def _test_image(seed: int = 0, n: int = 64) -> jnp.ndarray:
    """Deterministic random image."""
    return jax.random.normal(jax.random.PRNGKey(seed), (n, n))


class TestApertureFilter:
    """Aperture filter reproduces flux-map sampling."""

    def test_prepare_is_flux_map(self):
        """prepare() equals photometry.flux_map with the same kernel."""
        image = _test_image()
        kernel = make_aperture_kernel(radius=2.5, soft=True, sharpness=10.0)
        filt = ApertureFilter(kernel=kernel, order=3)
        np.testing.assert_array_equal(
            np.asarray(filt.prepare(image)), np.asarray(flux_map(image, kernel))
        )

    def test_evaluate_matches_map_coordinates(self):
        """evaluate() equals a direct map_coordinates sample."""
        image = _test_image(1)
        kernel = make_aperture_kernel(radius=2.5, soft=True, sharpness=10.0)
        filt = ApertureFilter(kernel=kernel, order=3)
        fmap = filt.prepare(image)
        pos = jnp.array([31.3, 40.7])
        expected = map_coordinates(fmap, jnp.array([[31.3], [40.7]]), order=3)[0]
        np.testing.assert_array_equal(
            np.asarray(filt.evaluate(fmap, pos)), np.asarray(expected)
        )

    def test_is_jit_and_vmap_compatible(self):
        """Filter passes through jit as a pytree and vmaps over positions."""
        image = _test_image(2)
        kernel = make_aperture_kernel(radius=2.0, soft=True, sharpness=10.0)
        filt = ApertureFilter(kernel=kernel, order=1)

        @eqx.filter_jit
        def run(f, img, positions):
            """Evaluate the filter at many positions."""
            fmap = f.prepare(img)
            return jax.vmap(lambda p: f.evaluate(fmap, p))(positions)

        out = run(filt, image, jnp.array([[20.0, 20.0], [30.5, 41.5]]))
        assert out.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(out)))


class TestGaussianFilter:
    """Gaussian filter reproduces the matched-filter preprocessing."""

    def test_kernel_matches_frozen_estimator(self):
        """gaussian_kernel_1d equals MatchedFilterSNREstimator's kernel."""
        est = MatchedFilterSNREstimator(fwhm=4.0)
        np.testing.assert_array_equal(
            np.asarray(gaussian_kernel_1d(4.0)), np.asarray(est.kernel_1d)
        )

    def test_prepare_uses_separable_convolution(self):
        """prepare() equals gaussian_filter_2d on the same kernel."""
        image = _test_image(3)
        kernel_1d = gaussian_kernel_1d(5.0)
        filt = GaussianFilter(kernel_1d=kernel_1d, order=3)
        np.testing.assert_array_equal(
            np.asarray(filt.prepare(image)),
            np.asarray(gaussian_filter_2d(image, kernel_1d)),
        )

    def test_filtered_image_smooths_noise(self):
        """Gaussian filtering reduces pixel-to-pixel variance."""
        image = _test_image(4)
        filtered = gaussian_filter_2d(image, gaussian_kernel_1d(5.0))
        assert float(jnp.std(filtered)) < float(jnp.std(image))
