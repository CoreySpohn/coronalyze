"""test_pipelines.py - Tests for yield simulation pipelines.

Run with: pytest tests/test_pipelines.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from coronalyze.core.modeling import subtract_disk, subtract_star
from coronalyze.pipelines import (
    calculate_yield_snr,
    klip_subtract,
)


class TestSubtractStar:
    """Tests for star subtraction primitive."""

    def test_exact_subtraction(self):
        """Should exactly subtract star model."""
        science = jnp.array([[10.0, 20.0], [30.0, 40.0]])
        star_model = jnp.array([[5.0, 10.0], [15.0, 20.0]])

        residual = subtract_star(science, star_model)
        expected = jnp.array([[5.0, 10.0], [15.0, 20.0]])

        assert jnp.allclose(residual, expected)

    def test_preserves_planet_signal(self):
        """Planet signal not in models should remain."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        star_model = 100 * jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 200)
        planet = 10 * jnp.exp(-((y - 40) ** 2 + (x - 50) ** 2) / 10)
        science = star_model + planet

        residual = subtract_star(science, star_model)

        assert jnp.allclose(residual, planet, atol=1e-5)


class TestSubtractDisk:
    """Tests for disk subtraction primitive."""

    def test_with_disk_model(self):
        """Should subtract disk model."""
        science = jnp.array([[100.0]])
        disk_model = jnp.array([[30.0]])

        residual = subtract_disk(science, disk_model)
        assert jnp.isclose(residual[0, 0], 70.0)


class TestKLIPSubtract:
    """Tests for KLIP subtraction pipeline."""

    def test_removes_common_structure(self):
        """KLIP should remove structure present in reference cube."""
        ny, nx = 32, 32
        n_frames = 10

        # Common structure
        y, x = jnp.mgrid[:ny, :nx]
        structure = jnp.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 50)

        # Reference cube = structure + noise
        key = jax.random.PRNGKey(0)
        ref_cube = structure[None, :, :] + 0.1 * jax.random.normal(
            key, (n_frames, ny, nx)
        )

        # Science = structure + planet
        planet = 0.5 * jnp.exp(-((y - 20) ** 2 + (x - 25) ** 2) / 10)
        science = structure + planet

        residual = klip_subtract(science, ref_cube, n_modes=3)

        # Planet should be preserved
        planet_region = residual[17:24, 22:28]
        assert jnp.max(planet_region) > 0.1


class TestCalculateYieldSNR:
    """Tests for end-to-end yield SNR calculation."""

    def test_star_method(self):
        """Test star method (formerly faux_rdi)."""
        ny, nx = 64, 64
        y, x = jnp.mgrid[:ny, :nx]

        star_model = 1000 * jnp.exp(-((y - 32) ** 2 + (x - 32) ** 2) / 100)
        planet_pos = (40.0, 45.0)
        planet = 50 * jnp.exp(
            -((y - planet_pos[0]) ** 2 + (x - planet_pos[1]) ** 2) / 5
        )

        # Add noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, (ny, nx)) * 2
        science = star_model + planet + noise

        # Calculate SNR
        positions = jnp.array([[40.0, 45.0]])
        snrs = calculate_yield_snr(
            science, positions, fwhm=4.0, star_model=star_model, method="star"
        )

        # Should detect planet
        assert snrs.shape == (1,)
        assert snrs[0] > 5  # Should be clearly detected

    def test_klip_method(self):
        """Test KLIP method through convenience wrapper."""
        ny, nx = 32, 32
        n_frames = 10

        key = jax.random.PRNGKey(1)
        ref_cube = jax.random.normal(key, (n_frames, ny, nx))
        science = jax.random.normal(jax.random.PRNGKey(2), (ny, nx))

        # Position must be at radius >= fwhm from center for valid SNR
        # Center is (15.5, 15.5), so (20, 20) gives radius ~ 6.4 > fwhm=3
        positions = jnp.array([[20.0, 20.0]])
        snrs = calculate_yield_snr(
            science,
            positions,
            fwhm=3.0,
            reference_cube=ref_cube,
            method="klip",
            n_modes=3,
        )

        assert snrs.shape == (1,)
        assert jnp.isfinite(snrs[0])

    def test_invalid_method_raises(self):
        """Should raise for unknown method."""
        science = jnp.zeros((32, 32))
        positions = jnp.array([[16.0, 16.0]])

        with pytest.raises(ValueError, match="Unknown method"):
            calculate_yield_snr(science, positions, fwhm=3.0, method="invalid")

    def test_missing_star_model_raises(self):
        """Star method should require star_model."""
        science = jnp.zeros((32, 32))
        positions = jnp.array([[16.0, 16.0]])

        with pytest.raises(ValueError, match="star_model required"):
            calculate_yield_snr(science, positions, fwhm=3.0, method="star")
