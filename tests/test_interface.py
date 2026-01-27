"""Tests for the coronagraphoto interface module."""

import jax.numpy as jnp
import pytest

from coronalyze.interfaces.coronagraphoto import (
    analyze_observation,
    extract_image,
    get_fwhm,
)


class TestExtractImage:
    """Tests for extract_image function."""

    def test_passthrough_2d(self):
        """2D images should pass through unchanged."""
        image = jnp.ones((50, 50))
        result = extract_image(image)
        assert result.shape == (50, 50)
        assert jnp.allclose(result, image)

    def test_extract_first_plane_from_3d(self):
        """3D datacubes should return first plane."""
        datacube = jnp.ones((10, 50, 50))
        result = extract_image(datacube)
        assert result.shape == (50, 50)


class TestGetFWHM:
    """Tests for get_fwhm function."""

    def test_fwhm_positive(self):
        """FWHM should be positive."""
        fwhm = get_fwhm(wavelength_nm=550.0)
        assert fwhm > 0

    def test_fwhm_scales_with_wavelength(self):
        """FWHM should increase with wavelength."""
        fwhm_blue = get_fwhm(wavelength_nm=400.0)
        fwhm_red = get_fwhm(wavelength_nm=800.0)
        assert fwhm_red > fwhm_blue

    def test_fwhm_scales_with_diameter(self):
        """FWHM should decrease with larger aperture."""
        fwhm_small = get_fwhm(wavelength_nm=550.0, diameter_m=4.0)
        fwhm_large = get_fwhm(wavelength_nm=550.0, diameter_m=8.0)
        assert fwhm_large < fwhm_small


class TestAnalyzeObservation:
    """Tests for analyze_observation convenience function."""

    def test_returns_snr(self):
        """Should return a finite SNR value."""
        shape = (51, 51)
        cy, cx = shape[0] / 2.0, shape[1] / 2.0

        # Create test image
        image = jnp.ones(shape) * 100.0
        planet_y, planet_x = cy + 10, cx

        y, x = jnp.ogrid[: shape[0], : shape[1]]
        sigma = 2.0
        planet = 500.0 * jnp.exp(
            -((y - planet_y) ** 2 + (x - planet_x) ** 2) / (2 * sigma**2)
        )
        image = image + planet

        snr = analyze_observation(
            image=image,
            planet_pos=(planet_y, planet_x),
            wavelength_nm=550.0,
        )

        assert jnp.isfinite(snr)
        assert snr > 0
