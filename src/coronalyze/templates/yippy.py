"""Yippy-backed template provider: coronagraph off-axis PSFs as templates."""

import equinox as eqx
import jax.numpy as jnp

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
    """Off-axis coronagraph PSF stamps from a yippy EqxCoronagraph.

    Maps a science-image pixel position to a (x, y) offset from the star in
    lambda/D, synthesizes the off-axis PSF there, and extracts a fixed-size
    stamp centered on the PSF's location in the coronagraph frame. When the
    science image shares the coronagraph grid (same pixel scale and
    orientation), the stamp is the expected planet signal at that pixel.

    The provider's center_yx must describe the same star center as the
    science frames it is used with.
    """

    coronagraph: EqxCoronagraph
    center_yx: jnp.ndarray
    stamp_size: int = eqx.field(static=True)
    lod_per_px: float = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __init__(
        self,
        coronagraph: EqxCoronagraph,
        center_yx,
        stamp_size: int,
        lod_per_px: float | None = None,
        order: int = 3,
    ):
        """Configure the provider.

        Args:
            coronagraph: Loaded yippy EqxCoronagraph.
            center_yx: (2,) star center (y, x) in science-image pixels.
            stamp_size: Stamp edge length in pixels.
            lod_per_px: Science-image pixel scale in lambda/D per pixel;
                None uses the coronagraph's own pixel scale (science grid
                equals the coronagraph grid).
            order: Interpolation order for stamp extraction.
        """
        self.coronagraph = coronagraph
        self.center_yx = jnp.asarray(center_yx)
        self.stamp_size = int(stamp_size)
        self.lod_per_px = (
            float(coronagraph.pixel_scale_lod)
            if lod_per_px is None
            else float(lod_per_px)
        )
        self.order = int(order)

    def __call__(self, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Synthesize the off-axis PSF stamp for one candidate position."""
        dy_px = position_yx[0] - self.center_yx[0]
        dx_px = position_yx[1] - self.center_yx[1]
        x_lod = dx_px * self.lod_per_px
        y_lod = dy_px * self.lod_per_px
        psf = self.coronagraph.create_psf(x_lod, y_lod)
        scale = self.coronagraph.pixel_scale_lod
        psf_center = jnp.array(
            [
                self.coronagraph.center_y + y_lod / scale,
                self.coronagraph.center_x + x_lod / scale,
            ]
        )
        return extract_patch(psf, psf_center, self.stamp_size, order=self.order)
