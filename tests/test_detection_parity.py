"""Golden parity: composed estimators reproduce the fused v1.1.1 cores."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from coronalyze.core.detection.estimator import DetectionEstimator
from coronalyze.core.detection.filters import (
    ApertureFilter,
    GaussianFilter,
    gaussian_kernel_1d,
)
from coronalyze.core.detection.samplers import AnnulusSampler, ApertureSampler
from coronalyze.core.detection.significance import AnnulusSigmaTest, TwoSampleTTest
from coronalyze.core.modeling import inject_planet
from coronalyze.core.photometry import make_aperture_kernel
from tests.golden_reference import (
    golden_matched_filter_snr_batch_core,
    golden_snr_batch_core,
)

FWHM = 5.0
MAX_APERTURES = 200
EXCLUSION_BUFFER = 0.5


def _gaussian_psf(n: int, fwhm: float = FWHM) -> jnp.ndarray:
    """Center-referenced Gaussian PSF template spanning the image."""
    c = (n - 1) / 2.0
    y, x = jnp.mgrid[:n, :n]
    sigma = fwhm / 2.355
    return jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))


def _scene(seed: int, n: int = 101) -> jnp.ndarray:
    """Speckled scene with two injected planets (the standard fixture)."""
    key = jax.random.PRNGKey(seed)
    image = 10.0 + 3.0 * jax.random.normal(key, (n, n))
    image = inject_planet(image, _gaussian_psf(n), 80.0, (50.0 + 18.0, 50.0))
    image = inject_planet(image, _gaussian_psf(n), 50.0, (50.0, 50.0 - 27.0))
    return image


def _positions(n: int = 101) -> jnp.ndarray:
    """Candidate grid: planets, empty field, small radius, off chip."""
    c = (n - 1) / 2.0
    return jnp.array(
        [
            [c + 18.0, c],
            [c, c - 27.0],
            [c - 22.0, c + 9.0],
            [c + 3.0, c],  # r < fwhm -> NaN via geometry gate
            [c + 2.0, c - 1.0],  # r < fwhm
            [c, c + 44.0],
            [c + 33.3, c - 12.7],
        ]
    )


def _mawet_composed() -> DetectionEstimator:
    """Composed estimator equivalent to SNREstimator(fwhm=FWHM)."""
    kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
    return DetectionEstimator(
        filter=ApertureFilter(kernel=kernel, order=3),
        sampler=ApertureSampler(
            fwhm=FWHM,
            max_apertures=MAX_APERTURES,
            exclusion_buffer=EXCLUSION_BUFFER,
        ),
        test=TwoSampleTTest(),
    )


def _assert_parity(actual, expected):
    """Bitwise equality first; document any relaxation (see plan protocol)."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if not np.array_equal(actual, expected, equal_nan=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


class TestMawetParity:
    """Composed Aperture+Aperture+TwoSampleT equals the fused Mawet core."""

    def test_statistic_parity_on_scene(self):
        """Statistics match the golden core on the standard scene."""
        image = _scene(0)
        positions = _positions()
        kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
        golden_snr, golden_n = golden_snr_batch_core(
            image, positions, kernel, FWHM, MAX_APERTURES, 3, EXCLUSION_BUFFER, None
        )
        stats = _mawet_composed()(image, positions)
        _assert_parity(stats.statistic, golden_snr)
        _assert_parity(stats.dof + 1.0, golden_n.astype(float))

    def test_statistic_parity_with_validity_map(self):
        """Validity-map threading matches the golden core."""
        image = _scene(1)
        positions = _positions()
        n = image.shape[0]
        validity = jnp.ones((n, n)).at[:, : n // 3].set(0.0)
        kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
        golden_snr, golden_n = golden_snr_batch_core(
            image, positions, kernel, FWHM, MAX_APERTURES, 3, EXCLUSION_BUFFER, validity
        )
        stats = _mawet_composed()(image, positions, validity)
        _assert_parity(stats.statistic, golden_snr)
        _assert_parity(stats.dof + 1.0, golden_n.astype(float))

    def test_statistic_parity_randomized(self):
        """Property sweep: random scenes and positions stay at parity."""
        kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
        composed = _mawet_composed()
        for seed in range(8):
            key = jax.random.PRNGKey(100 + seed)
            k1, k2 = jax.random.split(key)
            image = jax.random.normal(k1, (101, 101))
            r = jax.random.uniform(k2, (5,), minval=2.0, maxval=45.0)
            theta = jax.random.uniform(jax.random.PRNGKey(seed), (5,)) * 2 * jnp.pi
            positions = jnp.stack(
                [50.0 + r * jnp.sin(theta), 50.0 + r * jnp.cos(theta)], axis=1
            )
            golden_snr, _ = golden_snr_batch_core(
                image, positions, kernel, FWHM, MAX_APERTURES, 3, EXCLUSION_BUFFER, None
            )
            stats = composed(image, positions)
            _assert_parity(stats.statistic, golden_snr)

    def test_fpf_and_kind_populated(self):
        """The composed path adds finite FPF where the statistic is finite."""
        stats = _mawet_composed()(_scene(2), _positions())
        finite = ~jnp.isnan(stats.statistic)
        assert bool(jnp.all(jnp.isfinite(stats.fpf[finite])))
        assert bool(jnp.all(jnp.isnan(stats.fpf[~finite])))
        assert stats.statistic_kind == "two_sample_t"

    def test_jit_and_grad_through_composed(self):
        """The composed estimator jits and differentiates through the image."""
        composed = _mawet_composed()
        image = _scene(3)
        positions = _positions()[:2]

        @eqx.filter_jit
        def loss(img):
            """Negative statistic of the first candidate."""
            return -composed(img, positions).statistic[0]

        grad = jax.grad(loss)(image)
        assert grad.shape == image.shape
        assert bool(jnp.any(grad != 0.0))
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestMatchedFilterParity:
    """Composed Gaussian+Annulus+Sigma equals the fused matched-filter core."""

    def _composed(self, inner: float = -1.0, outer: float = -1.0):
        """Composed estimator equivalent to MatchedFilterSNREstimator."""
        return DetectionEstimator(
            filter=GaussianFilter(kernel_1d=gaussian_kernel_1d(FWHM), order=3),
            sampler=AnnulusSampler(fwhm=FWHM, annulus_inner=inner, annulus_outer=outer),
            test=AnnulusSigmaTest(),
        )

    def test_statistic_parity_on_scene(self):
        """Statistics match the golden matched-filter core (auto annulus)."""
        image = _scene(4)
        positions = _positions()
        golden = golden_matched_filter_snr_batch_core(
            image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, -1.0, -1.0
        )
        stats = self._composed()(image, positions)
        _assert_parity(stats.statistic, golden)
        assert bool(jnp.all(jnp.isinf(stats.dof)))

    def test_statistic_parity_explicit_annulus(self):
        """Explicit annulus bounds match the golden core."""
        image = _scene(5)
        positions = _positions()[:4]
        golden = golden_matched_filter_snr_batch_core(
            image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, 12.0, 20.0
        )
        stats = self._composed(12.0, 20.0)(image, positions)
        _assert_parity(stats.statistic, golden)

    def test_statistic_parity_randomized(self):
        """Property sweep for the matched-filter pairing."""
        composed = self._composed()
        for seed in range(6):
            image = jax.random.normal(jax.random.PRNGKey(200 + seed), (101, 101))
            r = jax.random.uniform(
                jax.random.PRNGKey(seed), (4,), minval=8.0, maxval=40.0
            )
            theta = (
                jax.random.uniform(jax.random.PRNGKey(300 + seed), (4,)) * 2 * jnp.pi
            )
            positions = jnp.stack(
                [50.0 + r * jnp.sin(theta), 50.0 + r * jnp.cos(theta)], axis=1
            )
            golden = golden_matched_filter_snr_batch_core(
                image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, -1.0, -1.0
            )
            stats = composed(image, positions)
            _assert_parity(stats.statistic, golden)


class TestCenterThreading:
    """center_yx=None equals an explicit geometric center, and shifts move it."""

    def test_none_equals_explicit_geometric(self):
        """Passing the geometric center explicitly is bit-identical to None."""
        image = _scene(6)
        positions = _positions()
        composed = _mawet_composed()
        c = (image.shape[0] - 1) / 2.0
        a = composed(image, positions)
        b = composed(image, positions, None, jnp.array([c, c]))
        np.testing.assert_array_equal(np.asarray(a.statistic), np.asarray(b.statistic))

    def test_offset_center_changes_statistics(self):
        """A shifted star center changes the reference geometry."""
        image = _scene(7)
        positions = _positions()[:3]
        composed = _mawet_composed()
        a = composed(image, positions)
        b = composed(image, positions, None, jnp.array([40.0, 55.0]))
        assert not np.array_equal(
            np.asarray(a.statistic), np.asarray(b.statistic), equal_nan=True
        )


class TestFrozenApiPinning:
    """Public frozen APIs reproduce the golden cores after delegation."""

    def test_snr_equals_golden_core(self):
        """snr() output is unchanged by the delegation refactor."""
        from coronalyze.core.photometry import make_aperture_kernel
        from coronalyze.core.snr import snr

        image = _scene(8)
        positions = _positions()
        kernel = make_aperture_kernel(radius=FWHM / 2.0, soft=True, sharpness=10.0)
        golden, _ = golden_snr_batch_core(
            image, positions, kernel, FWHM, MAX_APERTURES, 3, EXCLUSION_BUFFER, None
        )
        _assert_parity(snr(image, positions, FWHM), golden)

    def test_matched_filter_snr_equals_golden_core(self):
        """matched_filter_snr() output is unchanged by the delegation."""
        from coronalyze.core.matched_filter import matched_filter_snr

        image = _scene(9)
        positions = _positions()
        golden = golden_matched_filter_snr_batch_core(
            image, positions, FWHM, gaussian_kernel_1d(FWHM), 3, -1.0, -1.0
        )
        _assert_parity(matched_filter_snr(image, positions, FWHM), golden)
