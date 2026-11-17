"""Synthetic convention anchors and public-data yippy integration tests."""

import os
from dataclasses import replace

import equinox as eqx
import ineedvalidation as vv
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.grids import Frame, Image, PixelGrid, SkyOrientation, resample
from yippy import Coronagraph, fetch_yip
from yippy.datasets import cache_dir

from coronalyze.core.detection.estimator import DetectionEstimator
from coronalyze.core.detection.filters import PSFTemplateFilter
from coronalyze.core.detection.samplers import ApertureSampler
from coronalyze.core.detection.significance import TwoSampleTTest
from coronalyze.templates.yippy import YippyTemplateProvider

_CACHED = (cache_dir() / "eac1_aavc_2d.zip").exists()
_OPTED_IN = os.environ.get("CORONALYZE_FETCH_DATA") == "1"
STAMP_SIZE = 21
INJECTION_SEPARATION_LOD = 6.0
INJECTION_AMPLITUDE = 300.0


class SyntheticCoronagraph(eqx.Module):
    """Analytic off-axis source; field-dependent widths expose swapped offsets."""

    grid: PixelGrid

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid):
        """Return a Gaussian Image at the documented native-grid coordinate."""
        xy = self.grid.coordinates()
        x_lod, y_lod = x_lod[:, None, None], y_lod[:, None, None]
        return resample(
            Image(
                jnp.exp(
                    -0.5
                    * (
                        ((xy.x - x_lod) / (0.35 + 0.04 * x_lod)) ** 2
                        + ((xy.y - y_lod) / (0.30 + 0.04 * y_lod)) ** 2
                    )
                ),
                self.grid,
                ("source",),
                "per_pixel",
                "stellar_flux_fraction",
                reconstruction="cubic",
            ),
            grid,
        )


@vv.case("psf-template", "code-verification")
@vv.seam("yippy", "coronalyze")
@pytest.mark.parametrize("size", [40, 41])
@pytest.mark.parametrize(
    ("unit", "rotated"), [("lod", False), ("arcsec", False), ("arcsec", True)]
)
def test_peak_within_002_pixel_of_stamp_center(size, unit, rotated):
    """Fails if source offsets, grid center or row/column order change.

    Basis: an analytic Gaussian has its centroid at the source. A 0.02 px
    bound covers cubic/linear interpolation on these samples; a half-pixel
    mutant is 25 times larger. Field-dependent widths anchor input axes.
    Two shapes are required to distinguish even and odd center conventions.
    """
    frame = Frame(550.0, 6.0, "circumscribed")
    native = PixelGrid.regular((size, size), 0.25, "lod", frame=frame)
    if rotated:
        native = replace(native, orientation=SkyOrientation(0.0, -1))
    # A distinct science Frame makes copying native lambda/D sampling fail.
    science_frame = Frame(650.0, 4.0, "inscribed")
    step = 0.2 * 4.0 / 6.0
    # Deliberately rotate the science basis; +column points along native -y.
    science = PixelGrid(
        (size, size),
        (size - 1) / 2,
        (size - 1) / 2,
        ((0.0, step), (-step, 0.0)),
        "lod",
        frame=science_frame,
    )
    if unit == "arcsec":
        science = replace(
            science.to_arcsec(700.0),
            orientation=SkyOrientation(90.0, 1) if rotated else SkyOrientation(37.0),
        )
    provider = YippyTemplateProvider(
        SyntheticCoronagraph(native), science, 15, wavelength_nm=700.0
    )
    position = jnp.array([science.center_y - 4.0, science.center_x + 6.0])
    stamp = np.asarray(provider(position))
    y, x = np.indices(stamp.shape)
    centroid = np.array([(y * stamp).sum(), (x * stamp).sum()]) / stamp.sum()
    np.testing.assert_allclose(centroid, [7.0, 7.0], atol=0.02, rtol=0)
    # Known native source (-0.8,-1.2) lod: science x width=0.252/0.2,
    # science y width=0.318/0.2. These independent second moments reject
    # swapped source inputs even if extraction happens to recenter them.
    # With explicit orientations, E=-1.2,N=+0.8 and native x=+1.2,y=+0.8;
    # science row/column widths are then 0.332/0.2 and 0.398/0.2.
    moments = (
        np.array([((y - 7) ** 2 * stamp).sum(), ((x - 7) ** 2 * stamp).sum()])
        / stamp.sum()
    )
    expected = (
        [(0.332 / 0.2) ** 2, (0.398 / 0.2) ** 2]
        if rotated
        else [(0.318 / 0.2) ** 2, (0.252 / 0.2) ** 2]
    )
    np.testing.assert_allclose(moments, expected, atol=0.30, rtol=0)


@pytest.fixture(scope="session")
def coronagraph() -> Coronagraph:
    """Load the eac1 YIP coronagraph once per test session."""
    return Coronagraph(fetch_yip("eac1_aavc_2d"))


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with CORONALYZE_FETCH_DATA=1",
)
class TestStampShape:
    """The provider returns a fixed-size, finite stamp at real separations."""

    @pytest.mark.parametrize("separation_lod", [3.0, 6.0])
    def test_stamp_shape_and_finite(self, coronagraph, separation_lod):
        """The stamp is (STAMP_SIZE, STAMP_SIZE) and finite at the separation."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        dx_px = separation_lod / coronagraph.grid.step
        stamp = provider(center_yx + jnp.array([0.0, dx_px]))
        assert stamp.shape == (STAMP_SIZE, STAMP_SIZE)
        assert bool(jnp.all(jnp.isfinite(stamp)))


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with CORONALYZE_FETCH_DATA=1",
)
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
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        dy_px = dy_lod / coronagraph.grid.step
        dx_px = dx_lod / coronagraph.grid.step
        position_yx = center_yx + jnp.array([dy_px, dx_px])
        stamp = np.asarray(provider(position_yx))
        peak_yx = np.unravel_index(np.argmax(stamp), stamp.shape)
        stamp_center = (STAMP_SIZE - 1) / 2.0
        assert abs(peak_yx[0] - stamp_center) <= 1.0
        assert abs(peak_yx[1] - stamp_center) <= 1.0


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with CORONALYZE_FETCH_DATA=1",
)
class TestSeparationDependence:
    """Stamps at different separations are genuinely different templates."""

    def test_stamps_differ_after_peak_normalization(self, coronagraph):
        """Peak-normalized stamps at 2 and 7 lambda/D are not close."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        near = np.asarray(
            provider(center_yx + jnp.array([0.0, 2.0 / coronagraph.grid.step]))
        )
        far = np.asarray(
            provider(center_yx + jnp.array([0.0, 7.0 / coronagraph.grid.step]))
        )
        near = near / near.max()
        far = far / far.max()
        assert not np.allclose(near, far)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_CACHED or _OPTED_IN),
    reason="public YIP not cached; opt in with CORONALYZE_FETCH_DATA=1",
)
class TestEndToEnd:
    """The template filter detects an injected planet on the coronagraph grid."""

    def test_injected_planet_clears_detection_threshold(self, coronagraph):
        """An injected stamp at 6 lambda/D clears statistic 5 and fpf 1e-3."""
        center_yx = jnp.array([coronagraph.grid.center_y, coronagraph.grid.center_x])
        fwhm_px = 1.0 / coronagraph.grid.step
        provider = YippyTemplateProvider(
            coronagraph,
            coronagraph.grid,
            STAMP_SIZE,
            wavelength_nm=coronagraph.grid.frame.reference_wavelength_nm,
        )
        dx_px = INJECTION_SEPARATION_LOD / coronagraph.grid.step
        position_yx = center_yx + jnp.array([0.0, dx_px])

        stamp = provider(position_yx)
        half = STAMP_SIZE // 2
        y0, x0 = int(position_yx[0]) - half, int(position_yx[1]) - half

        image = jax.random.normal(jax.random.PRNGKey(0), coronagraph.grid.shape)
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
