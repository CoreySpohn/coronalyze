"""Yippy-backed template provider: coronagraph off-axis PSFs as templates."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
from hwoutils.grids import PixelGrid, resample

from coronalyze.core.detection.patches import extract_patch
from coronalyze.templates.base import AbstractTemplateProvider

try:
    from yippy import EqxCoronagraph
except ImportError as _err:  # pragma: no cover - exercised only without yippy
    raise ImportError(
        "YippyTemplateProvider requires yippy (>= 2.0); install the extra: "
        "pip install 'coronalyze[yippy]'"
    ) from _err


class YippyTemplateProvider(AbstractTemplateProvider):
    """Off-axis PSF stamps sampled on the science image's PixelGrid.

    Positions are (y, x) pixel-center coordinates. Source offsets are measured
    from each grid's center, through its complete basis and Frame. Both grids
    may declare sky orientation. An unoriented native grid shares the
    science detector axes; explicit orientations are transported through sky.
    The returned array is centered at ((stamp_size-1)/2, (stamp_size-1)/2).
    """

    coronagraph: EqxCoronagraph
    grid: PixelGrid
    stamp_size: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    _science_arcsec: PixelGrid
    _native_arcsec: PixelGrid
    _arcsec_per_lod: float = eqx.field(static=True)

    def __init__(
        self,
        coronagraph: EqxCoronagraph,
        grid: PixelGrid,
        stamp_size: int,
        *,
        wavelength_nm: float,
        order: int = 3,
    ):
        """Configure stamps for the supplied science grid and wavelength (nm).

        The coronagraph's lod grid defines create_psf offsets. Angular science
        sampling is converted with the native grid's Frame at wavelength_nm;
        a lod science grid uses its own Frame. No separate scale is accepted.
        order selects nearest (0), linear (1), or cubic (3) reconstruction
        for resampling and stamp extraction.
        """
        if coronagraph.grid.unit != "lod":
            raise ValueError("coronagraph grid must be in lod")
        if grid.unit not in ("arcsec", "lod"):
            raise ValueError("science grid must be in arcsec or lod")
        if grid.orientation is None and coronagraph.grid.orientation is not None:
            raise ValueError("science grid must declare the native sky orientation")
        if stamp_size < 1:
            raise ValueError("stamp_size must be positive")
        if order not in (0, 1, 3):
            raise ValueError("order must be 0, 1 or 3")
        self.coronagraph = coronagraph
        self.grid = grid
        self.stamp_size = int(stamp_size)
        self.order = int(order)
        native = coronagraph.grid.to_arcsec(wavelength_nm)
        self._native_arcsec = (
            replace(native, orientation=grid.orientation)
            if native.orientation is None
            else native
        )
        self._science_arcsec = (
            grid.to_arcsec(wavelength_nm) if grid.unit == "lod" else grid
        )
        self._arcsec_per_lod = float(
            coronagraph.grid.frame.lod_to_arcsec(wavelength_nm)
        )

    def __call__(self, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Synthesize an Image, resample its grid, and extract a centered stamp."""
        offset = self._science_arcsec.pixel_to_unit(position_yx[1], position_yx[0])
        x, y = offset.x, offset.y
        if self.grid.orientation != self._native_arcsec.orientation:
            sky = self.grid.orientation.detector_to_sky(x, y)
            native = self._native_arcsec.orientation.sky_to_detector(
                sky.east, sky.north
            )
            x, y = native.x, native.y
        psf = self.coronagraph.create_psf(
            x / self._arcsec_per_lod, y / self._arcsec_per_lod
        )
        angular_psf = replace(psf, grid=self._native_arcsec)
        sampled = resample(
            angular_psf,
            self._science_arcsec,
            reconstruction={0: "nearest", 1: "linear", 3: "cubic"}[self.order],
        )
        return extract_patch(
            sampled.data, position_yx, self.stamp_size, order=self.order
        )
