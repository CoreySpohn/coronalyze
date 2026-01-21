# coronalyze

**coronalyze** is a JAX-based Python library for post-processing coronagraphic direct imaging data. It is designed as a companion to **[coronagraphoto](https://github.com/CoreySpohn/coronagraphoto)**, providing analysis tools for simulated and real coronagraphic observations.

## Key Features

*   **Aperture Photometry**: Circular aperture photometry with throughput correction
*   **SNR Calculation**: Signal-to-noise ratio estimation for planet detection
*   **Count Rate Extraction**: Structured extraction of source count rates
*   **JAX-Native**: Fully JIT-compilable and GPU-accelerated

## Installation

```bash
pip install coronalyze
```

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements.)*

## Quick Start

```python
import coronalyze as cl

# Aperture photometry on a simulated image
flux = cl.aperture_photometry(image, position, radius_pixels=3)

# Calculate SNR
snr = cl.calculate_snr(planet_flux, background_noise, read_noise)
```

## Pairing with coronagraphoto

```python
import coronagraphoto as cg
import coronalyze as cl

# Generate image with coronagraphoto
scene = cg.load_sky_scene_from_exovista("system.fits")
image = simulate_exposure(scene, optical_path, exposure)

# Analyze with coronalyze
planet_flux = cl.aperture_photometry(image, planet_position, radius=3)
snr = cl.calculate_snr(planet_flux, noise_estimate)
```

## Design Philosophy

Like its companion `coronagraphoto`, `coronalyze` provides **primitives** rather than black-box functions. You compose the analysis pipeline that fits your science case.
