"""Integration tests for the yippy-backed template provider (data-gated)."""

import os

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yippy", reason="yippy extra not installed")

from coronalyze.datasets import PIKACHU, fetch_coronagraph

_CACHED = (PIKACHU.abspath / "coronagraphs.zip").exists()
_OPTED_IN = os.environ.get("CORONALYZE_FETCH_DATA") == "1"

pytestmark = pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="eac1 YIP not cached; set CORONALYZE_FETCH_DATA=1 to download",
)

import jax  # noqa: E402
from yippy import EqxCoronagraph  # noqa: E402

from coronalyze.core.detection.estimator import DetectionEstimator  # noqa: E402
from coronalyze.core.detection.filters import PSFTemplateFilter  # noqa: E402
from coronalyze.core.detection.samplers import ApertureSampler  # noqa: E402
from coronalyze.core.detection.significance import TwoSampleTTest  # noqa: E402
from coronalyze.templates.yippy import YippyTemplateProvider  # noqa: E402

STAMP_SIZE = 21
INJECTION_SEPARATION_LOD = 6.0
INJECTION_AMPLITUDE = 300.0


@pytest.fixture(scope="session")
def coronagraph() -> EqxCoronagraph:
    """Load the eac1 YIP coronagraph once per test session."""
    return EqxCoronagraph(fetch_coronagraph())


class TestStampShape:
    """The provider returns a fixed-size, finite stamp at real separations."""

    @pytest.mark.parametrize("separation_lod", [3.0, 6.0])
    def test_stamp_shape_and_finite(self, coronagraph, separation_lod):
        """The stamp is (STAMP_SIZE, STAMP_SIZE) and finite at the separation."""
        center_yx = jnp.array([coronagraph.center_y, coronagraph.center_x])
        provider = YippyTemplateProvider(coronagraph, center_yx, STAMP_SIZE)
        dx_px = separation_lod / coronagraph.pixel_scale_lod
        stamp = provider(center_yx + jnp.array([0.0, dx_px]))
        assert stamp.shape == (STAMP_SIZE, STAMP_SIZE)
        assert bool(jnp.all(jnp.isfinite(stamp)))


class TestOrientationPin:
    """The stamp peak lands where the pixel-to-lod mapping says it should.

    If this fails, the pixel-to-lod mapping or axis order in the provider is
    wrong; the fix belongs in the provider, not in a widened tolerance here.
    """

    @pytest.mark.parametrize(
        ("dy_lod", "dx_lod"),
        [(0.0, 5.0), (4.0, 0.0)],
    )
    def test_peak_within_one_pixel_of_stamp_center(self, coronagraph, dy_lod, dx_lod):
        """A stamp built for a +x or +y offset peaks within 1 px of its center."""
        center_yx = jnp.array([coronagraph.center_y, coronagraph.center_x])
        provider = YippyTemplateProvider(coronagraph, center_yx, STAMP_SIZE)
        dy_px = dy_lod / coronagraph.pixel_scale_lod
        dx_px = dx_lod / coronagraph.pixel_scale_lod
        position_yx = center_yx + jnp.array([dy_px, dx_px])
        stamp = np.asarray(provider(position_yx))
        peak_yx = np.unravel_index(np.argmax(stamp), stamp.shape)
        stamp_center = (STAMP_SIZE - 1) / 2.0
        assert abs(peak_yx[0] - stamp_center) <= 1.0
        assert abs(peak_yx[1] - stamp_center) <= 1.0


class TestSeparationDependence:
    """Stamps at different separations are genuinely different templates."""

    def test_stamps_differ_after_peak_normalization(self, coronagraph):
        """Peak-normalized stamps at 2 and 7 lambda/D are not close."""
        center_yx = jnp.array([coronagraph.center_y, coronagraph.center_x])
        provider = YippyTemplateProvider(coronagraph, center_yx, STAMP_SIZE)
        near = np.asarray(
            provider(center_yx + jnp.array([0.0, 2.0 / coronagraph.pixel_scale_lod]))
        )
        far = np.asarray(
            provider(center_yx + jnp.array([0.0, 7.0 / coronagraph.pixel_scale_lod]))
        )
        near = near / near.max()
        far = far / far.max()
        assert not np.allclose(near, far)


class TestEndToEnd:
    """The template filter detects an injected planet on the coronagraph grid."""

    def test_injected_planet_clears_detection_threshold(self, coronagraph):
        """An injected stamp at 6 lambda/D clears statistic 5 and fpf 1e-3."""
        center_yx = jnp.array([coronagraph.center_y, coronagraph.center_x])
        fwhm_px = 1.0 / coronagraph.pixel_scale_lod
        provider = YippyTemplateProvider(coronagraph, center_yx, STAMP_SIZE)
        dx_px = INJECTION_SEPARATION_LOD / coronagraph.pixel_scale_lod
        position_yx = center_yx + jnp.array([0.0, dx_px])

        stamp = provider(position_yx)
        half = STAMP_SIZE // 2
        y0, x0 = int(position_yx[0]) - half, int(position_yx[1]) - half

        image = jax.random.normal(jax.random.PRNGKey(0), coronagraph.psf_shape)
        image = image.at[y0 : y0 + STAMP_SIZE, x0 : x0 + STAMP_SIZE].add(
            INJECTION_AMPLITUDE * stamp
        )

        estimator = DetectionEstimator(
            filter=PSFTemplateFilter(provider=provider),
            sampler=ApertureSampler(fwhm=fwhm_px),
            test=TwoSampleTTest(),
        )
        stats = estimator(image, position_yx[None, :], None, center_yx)

        assert float(stats.statistic[0]) > 5.0
        assert float(stats.fpf[0]) < 1e-3
