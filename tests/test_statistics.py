"""Tests for coronalyze.core.statistics."""

import jax
import jax.numpy as jnp
import pytest

from coronalyze.core.statistics import (
    masked_mean,
    masked_std,
    student_t_sf,
)

# Reference values computed with scipy.stats.t.sf (scipy 1.16).
SCIPY_T_SF_CASES = [
    (0.0, 5.0, 0.50000000),
    (1.0, 5.0, 0.18160873),
    (2.5, 3.0, 0.04385332),
    (5.0, 2.0, 0.01887478),
    (5.0, 20.0, 0.00003437),
    (-1.5, 7.0, 0.91135076),
    (7.0, 6.0, 0.00021174),
]


@pytest.mark.parametrize(("t", "df", "expected"), SCIPY_T_SF_CASES)
def test_student_t_sf_matches_scipy_reference(t, df, expected):
    """Test student_t_sf against scipy reference values."""
    result = float(student_t_sf(jnp.asarray(t), jnp.asarray(df)))
    assert result == pytest.approx(expected, abs=1e-5)


def test_student_t_sf_is_vectorized():
    """Test that student_t_sf handles vectorized inputs."""
    t = jnp.array([0.0, 1.0, 2.5])
    df = jnp.array([5.0, 5.0, 3.0])
    result = student_t_sf(t, df)
    assert result.shape == (3,)
    assert float(result[0]) == pytest.approx(0.5, abs=1e-6)


def test_student_t_sf_monotone_decreasing_in_t():
    """Test that student_t_sf is monotone decreasing in t."""
    df = jnp.asarray(8.0)
    ts = jnp.linspace(-4.0, 6.0, 21)
    vals = student_t_sf(ts, df)
    assert bool(jnp.all(jnp.diff(vals) < 0.0))


def test_student_t_sf_invalid_dof_gives_nan():
    """Test that invalid degrees of freedom produce NaN."""
    assert bool(jnp.isnan(student_t_sf(jnp.asarray(2.0), jnp.asarray(0.0))))
    assert bool(jnp.isnan(student_t_sf(jnp.asarray(2.0), jnp.asarray(-3.0))))


def test_student_t_sf_propagates_nan_statistic():
    """Test that NaN in the statistic produces NaN output."""
    assert bool(jnp.isnan(student_t_sf(jnp.asarray(jnp.nan), jnp.asarray(5.0))))


def test_student_t_sf_jit_and_grad():
    """Test that student_t_sf is jitable and differentiable."""
    f = jax.jit(student_t_sf)
    val = f(jnp.asarray(2.0), jnp.asarray(10.0))
    assert jnp.isfinite(val)
    g = jax.grad(lambda t: student_t_sf(t, jnp.asarray(10.0)))(2.0)
    assert jnp.isfinite(g)
    assert g < 0.0  # survival function decreases in t


def test_masked_mean_ignores_masked_entries():
    """Test that masked_mean correctly ignores masked entries."""
    values = jnp.array([1.0, 2.0, 100.0])
    mask = jnp.array([1.0, 1.0, 0.0])
    assert float(masked_mean(values, mask)) == pytest.approx(1.5)


def test_masked_std_uses_bessel_correction():
    """Test that masked_std uses Bessel's correction (N-1 denominator)."""
    values = jnp.array([1.0, 3.0, 100.0])
    mask = jnp.array([1.0, 1.0, 0.0])
    # mean 2.0, residuals +/-1, variance = 2/(2-1) = 2
    assert float(masked_std(values, mask)) == pytest.approx(jnp.sqrt(2.0), rel=1e-6)
