"""Tests for the post-processing seam ABC and the Mawet arm."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from coronalyze.contracts import DetectionStats, FrameSet
from coronalyze.core.modeling import inject_planet
from coronalyze.core.snr import snr
from coronalyze.postproc import AbstractPostProcessing, MawetPostProcessing

FWHM_PX = 4.0


def make_frameset(image, n_frames=1, exposure_time_s=1.0):
    """Build a single-image FrameSet broadcast to n_frames identical frames."""
    frames = jnp.broadcast_to(image, (n_frames, *image.shape))
    return FrameSet(
        frames=frames,
        time_jd=jnp.arange(n_frames, dtype=float),
        exposure_time_s=jnp.full((n_frames,), exposure_time_s),
        telescope_pa_deg=jnp.zeros((n_frames,)),
        fwhm_px=FWHM_PX,
        wavelength_nm=550.0,
        bin_width_nm=110.0,
        pixel_scale_mas=21.8,
    )


def gaussian_psf(n=15, sigma=FWHM_PX / 2.355):
    """Build a normalized 2D Gaussian PSF template of size (n, n)."""
    y, x = jnp.mgrid[0:n, 0:n]
    c = (n - 1) / 2.0
    psf = jnp.exp(-((y - c) ** 2 + (x - c) ** 2) / (2 * sigma**2))
    return psf / jnp.sum(psf)


def test_abstract_postprocessing_cannot_instantiate():
    """Test that the abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        AbstractPostProcessing()


def test_mawet_arm_statistic_matches_snr_function():
    """Test that the Mawet arm's statistic matches the standalone snr() function."""
    key = jr.PRNGKey(0)
    image = jr.normal(key, (64, 64))
    positions = jnp.array([[32.0, 48.0], [16.0, 32.0]])
    fs = make_frameset(image)
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    stats = arm.detect(fs, positions)
    expected = snr(image, positions, fwhm=FWHM_PX)
    assert isinstance(stats, DetectionStats)
    assert jnp.allclose(stats.statistic, expected, equal_nan=True)
    assert stats.statistic_kind == "mawet_t"


def test_mawet_arm_bright_planet_has_small_fpf():
    """Test that a bright injected planet yields a high statistic and small FPF."""
    key = jr.PRNGKey(1)
    image = jr.normal(key, (64, 64))
    planet_pos = jnp.array([32.0, 50.0])
    image = inject_planet(image, gaussian_psf(n=64), flux=200.0, pos=planet_pos)
    fs = make_frameset(image)
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    stats = arm.detect(fs, planet_pos[None, :])
    assert float(stats.statistic[0]) > 5.0
    assert float(stats.fpf[0]) < 1e-3
    assert float(stats.dof[0]) >= 2.0


def test_mawet_arm_empty_field_fpf_is_calibrated():
    """Test that FPF values over an empty noise field are ~Uniform(0, 1)."""
    key = jr.PRNGKey(2)
    image = jr.normal(key, (128, 128))
    fs = make_frameset(image)
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    # 50 random positions in an annulus well inside the field
    pos_key, angle_key = jr.split(jr.PRNGKey(3))
    radii = jr.uniform(pos_key, (50,), minval=15.0, maxval=45.0)
    angles = jr.uniform(angle_key, (50,), minval=0.0, maxval=2 * jnp.pi)
    positions = jnp.stack(
        [63.5 + radii * jnp.sin(angles), 63.5 + radii * jnp.cos(angles)], axis=1
    )
    stats = arm.detect(fs, positions)
    finite = jnp.isfinite(stats.fpf)
    assert int(jnp.sum(finite)) > 40
    median_fpf = jnp.median(stats.fpf[finite])
    # Under H0 the FPF is ~Uniform(0, 1); its median should be central.
    assert 0.2 < float(median_fpf) < 0.8


def test_mawet_arm_nan_statistic_gives_nan_fpf():
    """Test that a too-close-in position gives a NaN statistic and NaN FPF."""
    key = jr.PRNGKey(4)
    image = jr.normal(key, (64, 64))
    fs = make_frameset(image)
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    stats = arm.detect(fs, jnp.array([[31.5, 32.5]]))  # r < fwhm -> NaN
    assert bool(jnp.isnan(stats.statistic[0]))
    assert bool(jnp.isnan(stats.fpf[0]))


def test_mawet_arm_ignores_references_and_uses_validity():
    """Test that the Mawet arm ignores references and honors science.validity."""
    key = jr.PRNGKey(5)
    image = jr.normal(key, (64, 64))
    validity = jnp.ones((64, 64)).at[:, :20].set(0.0)
    fs = eqx.tree_at(
        lambda f: f.validity,
        make_frameset(image),
        validity,
        is_leaf=lambda x: x is None,
    )
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    positions = jnp.array([[32.0, 48.0]])
    with_refs = arm.detect(fs, positions, references=make_frameset(image))
    without = arm.detect(fs, positions)
    assert jnp.allclose(with_refs.statistic, without.statistic, equal_nan=True)
    expected = snr(image, positions, fwhm=FWHM_PX, validity_map=validity)
    assert jnp.allclose(without.statistic, expected, equal_nan=True)


def test_mawet_arm_multiframe_uses_coadd():
    """Test that a multi-frame FrameSet is detected on its exposure-weighted coadd."""
    key = jr.PRNGKey(6)
    image = jr.normal(key, (64, 64)) + 10.0
    fs = make_frameset(image, n_frames=4, exposure_time_s=25.0)
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    positions = jnp.array([[32.0, 48.0]])
    stats = arm.detect(fs, positions)
    expected = snr(fs.coadd(), positions, fwhm=FWHM_PX)
    assert jnp.allclose(stats.statistic, expected, equal_nan=True)


def test_mawet_arm_detect_works_under_jit():
    """Test that arm.detect can be called from inside a jitted function."""
    key = jr.PRNGKey(7)
    image = jr.normal(key, (64, 64))
    fs = make_frameset(image)
    arm = MawetPostProcessing(fwhm_px=FWHM_PX)
    positions = jnp.array([[32.0, 48.0]])

    @jax.jit
    def run(f: FrameSet, p):
        return arm.detect(f, p).statistic

    assert jnp.allclose(run(fs, positions), arm.detect(fs, positions).statistic)
