"""Tests for SNREstimator.snr_and_dof (reference-aperture count exposure)."""

import jax.numpy as jnp
import jax.random as jr

from coronalyze.core.snr import snr_estimator


def make_image(key, n=64):
    """Generate a random Gaussian noise image for testing."""
    return jr.normal(key, (n, n))


def test_snr_and_dof_first_element_matches_call():
    """snr_and_dof's SNR values match __call__'s output exactly."""
    key = jr.PRNGKey(0)
    image = make_image(key)
    positions = jnp.array([[32.0, 45.0], [20.0, 20.0]])
    estimator = snr_estimator(fwhm=4.0)
    snr_only = estimator(image, positions)
    snr_vals, n_valid = estimator.snr_and_dof(image, positions)
    assert jnp.allclose(snr_only, snr_vals, equal_nan=True)
    assert n_valid.shape == (2,)


def test_n_valid_decreases_toward_small_separation():
    """Valid reference-aperture count shrinks at smaller separations."""
    key = jr.PRNGKey(1)
    image = make_image(key)
    estimator = snr_estimator(fwhm=4.0)
    # center is (31.5, 31.5); r ~ 6 px vs r ~ 25 px
    positions = jnp.array([[31.5, 37.5], [31.5, 56.5]])
    _, n_valid = estimator.snr_and_dof(image, positions)
    assert int(n_valid[0]) < int(n_valid[1])
    assert int(n_valid[0]) >= 3


def test_n_valid_respects_validity_map():
    """Masking half the field via validity_map reduces the valid count."""
    key = jr.PRNGKey(2)
    image = make_image(key)
    estimator = snr_estimator(fwhm=4.0)
    positions = jnp.array([[31.5, 51.5]])
    _, n_all = estimator.snr_and_dof(image, positions)
    validity = jnp.ones_like(image).at[:, :32].set(0.0)  # kill half the field
    _, n_masked = estimator.snr_and_dof(image, positions, validity)
    assert int(n_masked[0]) < int(n_all[0])
