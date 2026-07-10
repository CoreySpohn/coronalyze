"""Data contracts for coronalyze post-processing.

These types are the stable interchange surface between coronalyze and any
caller that renders frames (image simulators, mission simulators, real
pipelines). They are plain Equinox PyTrees over JAX arrays: callers construct
them from raw arrays; coronalyze algorithms consume and produce them.

Conventions:
    - Frames are detector electrons per frame (counts), shape
      (n_frames, ny, nx). Rate views divide by per-frame exposure time.
    - Field names carry unit suffixes (_jd, _s, _deg, _nm, _px, _mas).
    - Positions are (y, x) pixel coordinates ("_yx"), matching the image
      indexing convention used across the library.
"""

import equinox as eqx
import jax.numpy as jnp


def _asarray(value):
    return jnp.asarray(value, dtype=jnp.result_type(float))


def _asarray_or_none(value):
    return None if value is None else _asarray(value)


class FrameSet(eqx.Module):
    """A time series of detector frames plus the metadata detection needs.

    Attributes:
        frames: (n_frames, ny, nx) detector electrons per frame.
        time_jd: (n_frames,) frame start times, Julian date.
        exposure_time_s: (n_frames,) per-frame exposure times, seconds.
        telescope_pa_deg: (n_frames,) telescope position angle (roll), degrees.
        fwhm_px: Scalar resolution element (lambda/D) in pixels.
        wavelength_nm: Scalar band-center wavelength, nanometers.
        bin_width_nm: Scalar bandwidth, nanometers.
        pixel_scale_mas: Scalar detector pixel scale, milliarcseconds.
        noise_variance: Optional (n_frames, ny, nx) per-pixel total noise
            variance in electrons^2 (e.g. a detector model's variance budget);
            reserved for inverse-variance weighting (first consumer: the
            PSF-template matched filter).
        validity: Optional (ny, nx) mask (1 = usable pixel, 0 = excluded).
        center_yx: Optional (2,) star center (y, x); defaults to the geometric
            center ((ny - 1) / 2, (nx - 1) / 2) when None.
        tau_c_s: Optional speckle decorrelation time, seconds; scalar or
            (ny, nx) map. Reserved for correlated-noise-aware calibration.
    """

    frames: jnp.ndarray = eqx.field(converter=_asarray)
    time_jd: jnp.ndarray = eqx.field(converter=_asarray)
    exposure_time_s: jnp.ndarray = eqx.field(converter=_asarray)
    telescope_pa_deg: jnp.ndarray = eqx.field(converter=_asarray)
    fwhm_px: jnp.ndarray = eqx.field(converter=_asarray)
    wavelength_nm: jnp.ndarray = eqx.field(converter=_asarray)
    bin_width_nm: jnp.ndarray = eqx.field(converter=_asarray)
    pixel_scale_mas: jnp.ndarray = eqx.field(converter=_asarray)
    noise_variance: jnp.ndarray | None = eqx.field(
        default=None, converter=_asarray_or_none
    )
    validity: jnp.ndarray | None = eqx.field(default=None, converter=_asarray_or_none)
    center_yx: jnp.ndarray | None = eqx.field(default=None, converter=_asarray_or_none)
    tau_c_s: jnp.ndarray | None = eqx.field(default=None, converter=_asarray_or_none)

    def __check_init__(self):
        """Validate that field shapes are mutually consistent."""
        if self.frames.ndim != 3:
            raise ValueError(
                f"frames must be (n_frames, ny, nx); got shape {self.frames.shape}"
            )
        n_frames, ny, nx = self.frames.shape
        for name in ("time_jd", "exposure_time_s", "telescope_pa_deg"):
            value = getattr(self, name)
            if value.shape != (n_frames,):
                raise ValueError(
                    f"{name} must have shape ({n_frames},); got {value.shape}"
                )
        if self.noise_variance is not None and self.noise_variance.shape != (
            n_frames,
            ny,
            nx,
        ):
            raise ValueError(
                "noise_variance must match frames shape "
                f"({n_frames}, {ny}, {nx}); got {self.noise_variance.shape}"
            )
        if self.validity is not None and self.validity.shape != (ny, nx):
            raise ValueError(
                f"validity must have shape ({ny}, {nx}); got {self.validity.shape}"
            )
        if self.center_yx is not None and self.center_yx.shape != (2,):
            raise ValueError(
                f"center_yx must have shape (2,); got {self.center_yx.shape}"
            )

    @property
    def rate_frames(self) -> jnp.ndarray:
        """Per-frame count rates, electrons per second, (n_frames, ny, nx)."""
        return self.frames / self.exposure_time_s[:, None, None]

    def coadd(self) -> jnp.ndarray:
        """Exposure-weighted coadd: total electrons / total time, (ny, nx) e-/s."""
        return jnp.sum(self.frames, axis=0) / jnp.sum(self.exposure_time_s)


class DetectionStats(eqx.Module):
    """Per-candidate detection statistics produced by a post-processing arm.

    Every arm reports both the raw test statistic and its false positive
    fraction (FPF) under that arm's own null distribution, so downstream
    consumers can threshold on probability rather than raw sigma (a fixed
    sigma threshold is not a fixed false-alarm probability; Mawet et al.
    2014).

    Attributes:
        positions_yx: (n, 2) candidate positions, (y, x) pixels.
        statistic: (n,) test statistic values (NaN where unmeasurable).
        fpf: (n,) false positive fraction of each statistic under the arm's
            null distribution (NaN where the statistic is NaN).
        dof: (n,) effective degrees of freedom backing each statistic's null
            (t-tests: number of reference samples minus one).
        fwhm_px: Scalar resolution element used, pixels.
        statistic_kind: Static label of the statistic/null pairing, e.g.
            "mawet_t".
    """

    positions_yx: jnp.ndarray = eqx.field(converter=_asarray)
    statistic: jnp.ndarray = eqx.field(converter=_asarray)
    fpf: jnp.ndarray = eqx.field(converter=_asarray)
    dof: jnp.ndarray = eqx.field(converter=_asarray)
    fwhm_px: jnp.ndarray = eqx.field(converter=_asarray)
    statistic_kind: str = eqx.field(static=True)

    def __check_init__(self):
        """Validate that field shapes are mutually consistent."""
        if self.positions_yx.ndim != 2 or self.positions_yx.shape[-1] != 2:
            raise ValueError(
                f"positions_yx must be (n, 2); got {self.positions_yx.shape}"
            )
        n = self.positions_yx.shape[0]
        for name in ("statistic", "fpf", "dof"):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},); got {value.shape}")
