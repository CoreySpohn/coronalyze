"""test_yields.py - Tests for yield analysis tools.

Run with: pytest tests/test_yields.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from coronalyze.analysis.yields import (
    get_perfect_residuals,
    get_photon_noise_map,
    simulate_observation,
)


class TestGetPerfectResiduals:
    """Tests for perfect subtraction (Faux-RDI)."""

    def test_exact_subtraction(self):
        """Should return exact difference."""
        observation = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        model = jnp.array([[5.0, 10.0], [15.0, 20.0]])
        residual = get_perfect_residuals(observation, model)
        expected = jnp.array([[5.0, 10.0], [15.0, 20.0]])
        assert jnp.allclose(residual, expected)

    def test_preserves_planet_signal(self):
        """Planet signal not in model should remain in residual."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        # Background model (stellar halo)
        background = 100 * jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 200)

        # Planet signal
        planet_pos = (40, 50)
        planet = 10 * jnp.exp(
            -((y - planet_pos[0]) ** 2 + (x - planet_pos[1]) ** 2) / 10
        )

        observation = background + planet
        residual = get_perfect_residuals(observation, background)

        # Residual should contain only planet
        assert jnp.allclose(residual, planet, atol=1e-5)


class TestGetPhotonNoiseMap:
    """Tests for theoretical noise floor calculation."""

    def test_photon_dominated(self):
        """With no read noise, sigma = sqrt(flux*t)/t = sqrt(flux/t)."""
        rate = jnp.array([[100.0, 400.0], [900.0, 1600.0]])
        exposure_time = 1.0
        sigma = get_photon_noise_map(rate, exposure_time, read_noise=0.0)
        # sqrt(100*1)/1 = 10, sqrt(400*1)/1 = 20, etc
        expected = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        assert jnp.allclose(sigma, expected)

    def test_read_noise_contribution(self):
        """Read noise adds when flux is zero."""
        rate = jnp.array([[0.0]])
        exposure_time = 10.0
        read_noise = 10.0  # 10 electrons
        sigma = get_photon_noise_map(rate, exposure_time, read_noise=read_noise)
        # sqrt(0 + 10^2) / 10 = 1 e/s
        assert jnp.allclose(sigma, jnp.array([[1.0]]))

    def test_combined_noise(self):
        """Combined noise: sqrt(rate*t + RN^2) / t."""
        rate = jnp.array([[91.0]])  # 91*1 + 9 = 100
        exposure_time = 1.0
        read_noise = 3.0  # 3^2 = 9
        sigma = get_photon_noise_map(rate, exposure_time, read_noise=read_noise)
        # sqrt(91 + 9) / 1 = 10
        assert jnp.allclose(sigma, jnp.array([[10.0]]))

    def test_negative_flux_clipping(self):
        """Negative flux values should be clipped to zero."""
        rate = jnp.array([[-100.0, 100.0]])
        sigma = get_photon_noise_map(rate, exposure_time=1.0, read_noise=0.0)
        expected = jnp.array([[0.0, 10.0]])
        assert jnp.allclose(sigma, expected)


class TestSimulateObservation:
    """Tests for noisy observation simulation."""

    def test_returns_rate_units(self):
        """Output should be in rate units (count/sec)."""
        signal = jnp.zeros((32, 32))
        background = jnp.ones((32, 32)) * 100.0  # 100 counts/sec
        exposure_time = 10.0

        key = jax.random.PRNGKey(0)
        obs = simulate_observation(signal, background, exposure_time, key)

        # Mean should be close to background rate
        assert jnp.abs(jnp.mean(obs) - 100.0) < 10.0

    def test_poisson_statistics(self):
        """Noise should follow Poisson statistics."""
        n_trials = 1000
        background_rate = 100.0
        exposure_time = 1.0

        background = jnp.ones((1,)) * background_rate

        # Run many trials
        keys = jax.random.split(jax.random.PRNGKey(1), n_trials)
        observations = jax.vmap(
            lambda k: simulate_observation(
                jnp.zeros((1,)), background, exposure_time, k
            )
        )(keys)

        # Variance of Poisson should equal mean
        mean_obs = jnp.mean(observations)
        var_obs = jnp.var(observations)

        # For Poisson in rate units: var = mean / exposure_time
        expected_var = background_rate / exposure_time
        assert jnp.abs(var_obs - expected_var) / expected_var < 0.2

    def test_default_key(self):
        """Should work with default RNG key."""
        signal = jnp.zeros((32, 32))
        background = jnp.ones((32, 32)) * 100.0
        obs = simulate_observation(signal, background)
        assert obs.shape == (32, 32)
        assert jnp.all(jnp.isfinite(obs))

    def test_preserves_signal(self):
        """Planet signal should be present in observation."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        # Strong planet signal
        planet_pos = (32, 32)
        planet = 1000 * jnp.exp(
            -((y - planet_pos[0]) ** 2 + (x - planet_pos[1]) ** 2) / 10
        )
        background = jnp.ones((ny, nx)) * 10.0

        key = jax.random.PRNGKey(2)
        obs = simulate_observation(planet, background, exposure_time=1.0, rng_key=key)

        # Peak should still be at planet position
        peak_pos = jnp.unravel_index(jnp.argmax(obs), obs.shape)
        assert peak_pos == planet_pos
