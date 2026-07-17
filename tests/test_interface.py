"""Tests for the coronagraphoto interface module."""

import subprocess
import sys

import jax.numpy as jnp

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


def test_seam_contract_names_resolve_at_top_level():
    """Downstream registries resolve these exact attribute paths by import."""
    import coronalyze

    for name in (
        "FrameSet",
        "DetectionStats",
        "AbstractPostProcessing",
        "MawetPostProcessing",
        "student_t_sf",
    ):
        assert hasattr(coronalyze, name), name
        assert name in coronalyze.__all__, name


def test_detection_core_names_resolve_at_top_level():
    """Phase 2 detection names are importable from the package root."""
    import coronalyze

    for name in (
        "DetectionEstimator",
        "AbstractFilter",
        "AbstractSampler",
        "AbstractTest",
        "ApertureFilter",
        "GaussianFilter",
        "ApertureSampler",
        "AnnulusSampler",
        "TwoSampleTTest",
        "AnnulusSigmaTest",
        "GrubbsTest",
        "normal_sf",
        "grubbs_fpf",
        "n_reference_apertures",
        "matched_filter_snr",
        "matched_filter_snr_estimator",
        "MatchedFilterSNREstimator",
        "gaussian_kernel_1d",
        "gaussian_filter_2d",
    ):
        assert hasattr(coronalyze, name), name
        assert name in coronalyze.__all__, name


def test_template_filter_names_resolve_at_top_level():
    """PSF-template filter and template-provider names resolve at the package root."""
    import coronalyze

    for name in (
        "PSFTemplateFilter",
        "extract_patch",
        "AbstractTemplateProvider",
        "ArrayTemplateProvider",
        "MatchedFilterPostProc",
    ):
        assert hasattr(coronalyze, name), name
        assert name in coronalyze.__all__, name


def test_plain_import_does_not_load_yippy_provider():
    """Importing coronalyze must not pull in the yippy-backed template provider.

    coronalyze.templates.yippy imports yippy at module scope, so it has to
    stay unimported until something explicitly reaches for it; that is what
    keeps the base install yippy-free. Checked in a fresh subprocess so an
    unrelated test that already imported templates.yippy cannot mask a
    regression here.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, coronalyze; "
            "assert 'coronalyze.templates.yippy' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
