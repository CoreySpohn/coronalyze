"""Tests for the coronalyze seam contract types (FrameSet, DetectionStats)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from coronalyze.contracts import FrameSet


def make_frameset(n_frames=3, ny=8, nx=8, **overrides):
    """Build a FrameSet with valid defaults, applying any overrides."""
    kwargs = dict(
        frames=jnp.ones((n_frames, ny, nx)),
        time_jd=jnp.arange(n_frames, dtype=float),
        exposure_time_s=jnp.full((n_frames,), 100.0),
        telescope_pa_deg=jnp.zeros((n_frames,)),
        fwhm_px=3.0,
        wavelength_nm=550.0,
        bin_width_nm=110.0,
        pixel_scale_mas=21.8,
    )
    kwargs.update(overrides)
    return FrameSet(**kwargs)


def test_frameset_constructs_with_required_fields():
    """Test construction with required fields only; optionals default to None."""
    fs = make_frameset()
    assert fs.frames.shape == (3, 8, 8)
    assert fs.noise_variance is None
    assert fs.validity is None
    assert fs.center_yx is None
    assert fs.tau_c_s is None


def test_frameset_converts_inputs_to_jax_arrays():
    """Test that scalar and array inputs are converted to JAX arrays."""
    fs = make_frameset(fwhm_px=3.0)
    assert isinstance(fs.fwhm_px, jnp.ndarray)
    assert isinstance(fs.frames, jnp.ndarray)


def test_frameset_rejects_wrong_frames_ndim():
    """Test that frames without a (n_frames, ny, nx) shape raise ValueError."""
    with pytest.raises(ValueError, match="frames"):
        make_frameset(frames=jnp.ones((8, 8)))


def test_frameset_rejects_mismatched_per_frame_metadata():
    """Test that per-frame metadata of the wrong length raises ValueError."""
    with pytest.raises(ValueError, match="time_jd"):
        make_frameset(time_jd=jnp.arange(2, dtype=float))


def test_frameset_rejects_mismatched_noise_variance_shape():
    """Test that noise_variance not matching frames shape raises ValueError."""
    with pytest.raises(ValueError, match="noise_variance"):
        make_frameset(noise_variance=jnp.ones((3, 4, 4)))


def test_frameset_rate_frames_divides_by_exposure():
    """Test that rate_frames divides counts by per-frame exposure time."""
    fs = make_frameset()
    assert fs.rate_frames.shape == (3, 8, 8)
    assert float(fs.rate_frames[0, 0, 0]) == pytest.approx(0.01)


def test_frameset_coadd_is_total_electrons_over_total_time():
    """Test that coadd returns total electrons over total exposure time."""
    frames = jnp.stack([jnp.full((4, 4), 10.0), jnp.full((4, 4), 30.0)])
    fs = make_frameset(
        n_frames=2,
        ny=4,
        nx=4,
        frames=frames,
        time_jd=jnp.array([0.0, 1.0]),
        exposure_time_s=jnp.array([100.0, 300.0]),
        telescope_pa_deg=jnp.zeros((2,)),
    )
    coadd = fs.coadd()
    assert coadd.shape == (4, 4)
    assert float(coadd[0, 0]) == pytest.approx(40.0 / 400.0)


def test_frameset_is_a_pytree():
    """Test that FrameSet flattens and unflattens as a JAX pytree."""
    fs = make_frameset()
    leaves, treedef = jax.tree_util.tree_flatten(fs)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, FrameSet)
    assert jnp.array_equal(rebuilt.frames, fs.frames)


def test_frameset_passes_through_jit():
    """Test that a FrameSet can be passed through a jitted function."""
    fs = make_frameset()

    @jax.jit
    def total_electrons(f: FrameSet):
        return jnp.sum(f.frames)

    assert float(total_electrons(fs)) == pytest.approx(3 * 8 * 8)


def test_frameset_equinox_partition_roundtrip():
    """Test that eqx.partition and eqx.combine round-trip a FrameSet."""
    fs = make_frameset()
    params, static = eqx.partition(fs, eqx.is_array)
    rebuilt = eqx.combine(params, static)
    assert jnp.array_equal(rebuilt.frames, fs.frames)
