"""The composed detection estimator: filter + sampler + significance test."""

import equinox as eqx
import jax
import jax.numpy as jnp

from coronalyze.contracts import DetectionStats
from coronalyze.core.detection.base import AbstractFilter, AbstractSampler, AbstractTest


class DetectionEstimator(eqx.Module):
    """Composable detection: any filter, any sampler, any significance test.

    All three components are Equinox modules, so the composed estimator is a
    single pytree and the whole pipeline fuses into one XLA kernel under jit;
    the decomposition is purely a source-level concern. The estimator
    resolves defaults (all-valid map, geometric star center), vmaps the
    per-candidate computation, and NaNs statistic and FPF where the sampler
    reports the geometry invalid.
    """

    filter: AbstractFilter
    sampler: AbstractSampler
    test: AbstractTest

    @eqx.filter_jit
    def __call__(
        self,
        image: jnp.ndarray,
        positions_yx: jnp.ndarray,
        validity_map: jnp.ndarray | None = None,
        center_yx: jnp.ndarray | None = None,
    ) -> DetectionStats:
        """Detection statistics at candidate positions.

        Args:
            image: 2D science image.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            validity_map: Optional (ny, nx) mask (1 = usable, 0 = excluded).
            center_yx: Optional (2,) star center; geometric center when None.

        Returns:
            DetectionStats for the n candidates (statistic_kind from the
            test, fwhm_px from the sampler).
        """
        ny, nx = image.shape
        filtered = self.filter.prepare(image)
        if validity_map is None:
            validity_map = jnp.ones((ny, nx))
        if center_yx is None:
            center = jnp.array([(ny - 1) / 2.0, (nx - 1) / 2.0])
        else:
            center = jnp.asarray(center_yx)

        def _single(position_yx: jnp.ndarray):
            """Statistic, FPF, and dof for one candidate."""
            filt = self.filter.bind(position_yx)
            signal = filt.evaluate(filtered, position_yx)
            samples, mask, geom_ok = self.sampler(
                filtered, position_yx, filt, validity_map, center
            )
            statistic, fpf, dof = self.test(signal, samples, mask)
            statistic = jnp.where(geom_ok, statistic, jnp.nan)
            fpf = jnp.where(geom_ok, fpf, jnp.nan)
            return statistic, fpf, dof

        statistic, fpf, dof = jax.vmap(_single)(positions_yx)
        return DetectionStats(
            positions_yx=positions_yx,
            statistic=statistic,
            fpf=fpf,
            dof=dof,
            fwhm_px=self.sampler.fwhm,
            statistic_kind=self.test.kind,
        )
