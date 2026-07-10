"""Abstract interfaces for the filter / sampler / significance decomposition.

Every image-domain detection algorithm factors into three composable steps:
a filter measuring "how much signal is here", a sampler collecting "what
does noise look like here", and a significance test converting the two into
a statistic with a false positive fraction under its own null. All three
are Equinox modules so a composed estimator JIT-compiles into a single
fused kernel.
"""

import abc

import equinox as eqx
import jax.numpy as jnp


class AbstractFilter(eqx.Module):
    """Signal-extraction step: per-image precompute plus point evaluation."""

    @abc.abstractmethod
    def prepare(self, image: jnp.ndarray) -> jnp.ndarray:
        """Per-image precomputation (e.g. a flux map or filtered image).

        Args:
            image: 2D science image.

        Returns:
            The prepared response map evaluate() samples from.
        """

    @abc.abstractmethod
    def evaluate(self, filtered: jnp.ndarray, position_yx: jnp.ndarray) -> jnp.ndarray:
        """Scalar filter response at one (y, x) position.

        Args:
            filtered: Output of prepare() for the current image.
            position_yx: (2,) position in pixels.

        Returns:
            Scalar response.
        """


class AbstractSampler(eqx.Module):
    """Noise-reference collection step for a candidate position."""

    fwhm: eqx.AbstractVar[float]

    @abc.abstractmethod
    def __call__(
        self,
        filtered: jnp.ndarray,
        position_yx: jnp.ndarray,
        filt: AbstractFilter,
        validity_map: jnp.ndarray,
        center_yx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Collect noise reference samples for one candidate position.

        Args:
            filtered: Output of filt.prepare() for the current image.
            position_yx: (2,) candidate position in pixels.
            filt: The filter whose response defines the samples (evaluated
                with the candidate's own configuration; see the sliding-
                template rule in the architecture).
            validity_map: (ny, nx) map, 1 = usable pixel, 0 = excluded.
            center_yx: (2,) star center in pixels.

        Returns:
            Tuple of (samples, mask, geom_ok): a static-size sample buffer,
            its validity mask, and a scalar bool for geometry-level validity
            of the candidate itself.
        """


class AbstractTest(eqx.Module):
    """Significance step: signal plus noise samples to (statistic, fpf, dof)."""

    kind: eqx.AbstractClassVar[str]

    @abc.abstractmethod
    def __call__(
        self,
        signal: jnp.ndarray,
        samples: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Convert a signal measurement and noise samples into significance.

        Args:
            signal: Scalar filter response at the candidate.
            samples: Noise reference sample buffer from the sampler.
            mask: Boolean validity mask over samples.

        Returns:
            Tuple of scalars (statistic, fpf, dof); the statistic is NaN
            when the test's own sample-sufficiency gate fails, fpf follows
            the statistic, and dof is reported ungated.
        """
