"""test_validation.py - Geometric and physical validation tests.

Validates geometric conventions, centering, and unit consistency.
Run with: pytest tests/test_validation.py -v
"""

import jax.numpy as jnp
import pytest

from coronalyze.analysis.yields import get_photon_noise_map
from coronalyze.core.image_transforms import resample_flux
from coronalyze.core.modeling import inject_planet, make_simple_disk


class TestRotationDirection:
    """Verify rotation follows coronagraphoto convention."""

    def test_rotation_matches_coronagraphoto(self):
        """Positive rotation should match coronagraphoto's behavior.

        Note: Due to inverse mapping, positive rotation_deg rotates
        the image content CW. This matches coronagraphoto exactly.
        """
        # Create image with a dot on the Right (East, +X)
        img = jnp.zeros((33, 33))
        img = img.at[16, 26].set(1.0)  # Center at (16,16), dot at x=+10

        # Apply rotation - should match coronagraphoto
        rotated = resample_flux(img, 1.0, 1.0, (33, 33), rotation_deg=90.0)

        # Due to inverse mapping, +90 deg CCW matrix => image rotates CW
        # Dot at Right (+X) moves to Bottom (higher Y index)
        peak = jnp.unravel_index(jnp.argmax(rotated), rotated.shape)
        # Peak should be at y=26 (Bottom), x=16 (Center) for CW rotation
        assert peak == (26, 16), f"Expected (26,16), got {peak}"


class TestInjectionCentering:
    """Verify planet injection uses correct geometric center."""

    def test_center_injection_no_shift(self):
        """Injecting at geometric center should cause zero shift."""
        # 11x11 image. Geometric center is index 5.0 (0-indexed)
        img = jnp.zeros((11, 11))
        psf = jnp.zeros((11, 11))
        psf = psf.at[5, 5].set(1.0)

        # Inject at exact geometric center (5.0, 5.0)
        result = inject_planet(img, psf, flux=1.0, pos=(5.0, 5.0))

        # Peak should remain exactly at (5, 5) with value 1.0
        peak_pos = jnp.unravel_index(jnp.argmax(result), result.shape)
        assert peak_pos == (5, 5), f"Expected (5, 5), got {peak_pos}"
        assert jnp.isclose(jnp.max(result), 1.0, atol=1e-4)

    def test_even_sized_image_center(self):
        """Even-sized image: center is between pixels."""
        # 10x10 image. Geometric center is 4.5
        img = jnp.zeros((10, 10))
        psf = jnp.zeros((10, 10))
        # PSF at (4, 4) and (5, 5) equally weighted
        psf = psf.at[4, 4].set(0.5)
        psf = psf.at[5, 5].set(0.5)

        # Inject at center (4.5, 4.5) - should not shift
        result = inject_planet(img, psf, flux=1.0, pos=(4.5, 4.5))

        # Both peaks should be preserved
        assert result[4, 4] > 0.4
        assert result[5, 5] > 0.4


class TestDiskOrientation:
    """Verify disk position angle follows astronomical convention."""

    def test_pa_zero_is_north(self):
        """PA=0 should produce a vertical (North-South) disk."""
        shape = (101, 101)
        # Edge-on disk (needle) with PA=0
        disk = make_simple_disk(shape, radius=20, inclination_deg=85, width=2, pa_deg=0)

        # Calculate variance along each axis relative to center
        y, x = jnp.indices(shape)
        center = 50
        total = jnp.sum(disk)
        var_y = jnp.sum(disk * (y - center) ** 2) / total
        var_x = jnp.sum(disk * (x - center) ** 2) / total

        # If PA=0 is North (vertical), variance in Y > variance in X
        assert (
            var_y > var_x
        ), f"PA=0 produced horizontal disk (var_y={var_y:.1f}, var_x={var_x:.1f})"

    def test_pa_90_is_east(self):
        """PA=90 should produce a horizontal (East-West) disk."""
        shape = (101, 101)
        disk = make_simple_disk(
            shape, radius=20, inclination_deg=85, width=2, pa_deg=90
        )

        y, x = jnp.indices(shape)
        center = 50
        total = jnp.sum(disk)
        var_y = jnp.sum(disk * (y - center) ** 2) / total
        var_x = jnp.sum(disk * (x - center) ** 2) / total

        # If PA=90 is East (horizontal), variance in X > variance in Y
        assert (
            var_x > var_y
        ), f"PA=90 produced vertical disk (var_y={var_y:.1f}, var_x={var_x:.1f})"


class TestNoiseMapUnits:
    """Verify noise map handles rate vs counts correctly."""

    def test_photon_noise_rate_units(self):
        """Rate=100 e/s, t=1s, RN=0 => sigma=10 e/s."""
        sigma = get_photon_noise_map(
            expectation_rate=jnp.array([[100.0]]),
            exposure_time=1.0,
            read_noise=0.0,
        )
        assert jnp.isclose(sigma[0, 0], 10.0)

    def test_read_noise_scaling_with_time(self):
        """Rate=0, t=10s, RN=10e => sigma=1 e/s (RN dominates)."""
        sigma = get_photon_noise_map(
            expectation_rate=jnp.array([[0.0]]),
            exposure_time=10.0,
            read_noise=10.0,
        )
        # sqrt(0 + 100) / 10 = 1
        assert jnp.isclose(sigma[0, 0], 1.0)

    def test_longer_exposure_reduces_rate_noise(self):
        """Longer exposure should reduce noise in rate units."""
        rate = jnp.array([[100.0]])

        sigma_1s = get_photon_noise_map(rate, exposure_time=1.0)
        sigma_100s = get_photon_noise_map(rate, exposure_time=100.0)

        # sqrt(100*1)/1 = 10, sqrt(100*100)/100 = 1
        assert jnp.isclose(sigma_1s[0, 0], 10.0)
        assert jnp.isclose(sigma_100s[0, 0], 1.0)
