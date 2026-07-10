"""The post-processing seam: an abstract interface plus concrete arms.

A post-processing arm consumes a FrameSet (science frames plus metadata,
optionally with a reference FrameSet) and produces DetectionStats
(per-candidate statistic, false positive fraction, and degrees of freedom).
Downstream mission simulators register arms against AbstractPostProcessing
and thread FrameSet/DetectionStats through their pipelines; this module owns
the vocabulary and must stay import-clean of any downstream package.
"""

import abc

import equinox as eqx
import jax.numpy as jnp

from coronalyze.contracts import DetectionStats, FrameSet
from coronalyze.core.detection.estimator import DetectionEstimator
from coronalyze.core.detection.filters import ApertureFilter
from coronalyze.core.detection.samplers import ApertureSampler
from coronalyze.core.detection.significance import TwoSampleTTest
from coronalyze.core.snr import SNREstimator


class AbstractPostProcessing(eqx.Module):
    """Abstract post-processing arm: FrameSet -> DetectionStats.

    Implementations are Equinox modules (PyTrees), constructed per
    configuration (e.g. per band, with a concrete resolution element), so a
    single instance JIT-compiles cleanly inside larger pipelines.
    """

    @abc.abstractmethod
    def detect(
        self,
        science: FrameSet,
        positions_yx: jnp.ndarray,
        *,
        references: FrameSet | None = None,
    ) -> DetectionStats:
        """Compute detection statistics at candidate positions.

        Args:
            science: Science frames plus metadata.
            positions_yx: (n, 2) candidate positions, (y, x) pixels.
            references: Optional reference frames (for arms that difference
                against a reference library). Arms that self-reference may
                ignore it.

        Returns:
            DetectionStats for the n candidates.
        """


class MawetPostProcessing(AbstractPostProcessing):
    """Aperture-photometry detection on the coadded frame (Mawet et al. 2014).

    The statistic is the small-sample-corrected two-sample t-statistic of the
    candidate aperture against same-radius reference apertures; the FPF is
    its Student-t upper tail with (n_references - 1) degrees of freedom. The
    statistic is scale-invariant, so it is computed on the exposure-weighted
    coadd regardless of frame count. References are ignored: the noise sample
    comes from the science image itself.
    FrameSet.center_yx is consulted: reference-aperture geometry is computed
    about it, with the geometric image center used when it is None.
    """

    estimator: SNREstimator

    def __init__(
        self,
        fwhm_px: float,
        soft: bool = True,
        sharpness: float = 10.0,
        fast: bool = False,
        max_apertures: int = 200,
        exclusion_buffer: float = 0.5,
    ):
        """Build the arm with a concrete resolution element.

        Args:
            fwhm_px: Resolution element (lambda/D) in pixels; must match the
                FrameSets this arm is applied to.
            soft: Use differentiable soft aperture edges.
            sharpness: Sigmoid sharpness for soft apertures.
            fast: Bilinear (True) vs cubic (False) sub-pixel sampling.
            max_apertures: Static reference-aperture buffer size.
            exclusion_buffer: Gap between test and first reference aperture,
                in angular-step units.
        """
        self.estimator = SNREstimator(
            fwhm=fwhm_px,
            soft=soft,
            sharpness=sharpness,
            fast=fast,
            max_apertures=max_apertures,
            exclusion_buffer=exclusion_buffer,
        )

    def detect(
        self,
        science: FrameSet,
        positions_yx: jnp.ndarray,
        *,
        references: FrameSet | None = None,
    ) -> DetectionStats:
        """Compute the Mawet t-statistic and FPF on the science coadd."""
        del references  # Mawet self-references; see class docstring.
        image = science.coadd()
        composed = DetectionEstimator(
            filter=ApertureFilter(
                kernel=self.estimator.kernel, order=self.estimator.order
            ),
            sampler=ApertureSampler(
                fwhm=self.estimator.fwhm,
                max_apertures=self.estimator.max_apertures,
                exclusion_buffer=self.estimator.exclusion_buffer,
            ),
            test=TwoSampleTTest(),
        )
        stats = composed(image, positions_yx, science.validity, science.center_yx)
        return DetectionStats(
            positions_yx=stats.positions_yx,
            statistic=stats.statistic,
            fpf=stats.fpf,
            dof=stats.dof,
            fwhm_px=stats.fwhm_px,
            statistic_kind="mawet_t",
        )
