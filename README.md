<p align="center">
  <a href="https://pypi.org/project/coronablink/"><img src="https://img.shields.io/pypi/v/coronablink.svg?style=flat-square" alt="PyPI"/></a>
  <a href="https://coronablink.readthedocs.io"><img src="https://readthedocs.org/projects/coronablink/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <!-- <a href="https://github.com/CoreySpohn/coronablink/actions/workflows/ci.yml/"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/coronablink/ci.yml?branch=main&logo=github&style=flat-square" alt="CI"/></a> -->
</p>

---

# coronablink

**coronablink** is a JAX-based Python library for post-processing coronagraphic direct imaging data. It is designed as a companion to **[coronagraphoto](https://github.com/CoreySpohn/coronagraphoto)**, providing analysis tools for simulated and (maybe) real coronagraphic observations.

*The name references the [blink comparator](https://en.wikipedia.org/wiki/Blink_comparator), the instrument used to discover Pluto by quickly switching between two photographic plates (which is cool and vaguely similar to RDI). Also it's supposed to be fast, which is why it's written in JAX.*

## Key Features

*   **SNR Estimation**: Mawet et al. (2014) aperture photometry SNR with small-sample corrections
*   **PSF Subtraction**: KLIP/PCA and reference differential imaging
*   **Planet Injection**: Forward modeling for yield simulations
*   **JAX-Native**: Fully JIT-compilable, differentiable, and GPU-accelerated

## Installation

```bash
pip install coronablink
```

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements.)*

## Quick Start

```python
import coronablink as cb
import jax.numpy as jnp

# Calculate SNR at planet positions using Mawet et al. (2014)
snr_values = cb.snr(residual_image, planet_positions, fwhm=4.5)

# Or use the estimator pattern for efficient batch processing
estimator = cb.snr_estimator(fwhm=4.5)
snr_values = estimator(residual_image, planet_positions)
```

## Pairing with coronagraphoto

```python
import coronagraphoto as cg
import coronablink as cb

# Generate image with coronagraphoto
obs = cg.Observation(scene, coronagraph, settings)
obs.create_images()
science_image = obs.get_total_image()

# Get the star model for subtraction
star_model = obs.get_star_image()

# Subtract and calculate SNR
residual = cb.subtract_star(science_image, star_model)
snr_values = cb.snr(residual, planet_positions, fwhm=4.5)
```

## Design Philosophy

Like its companion `coronagraphoto`, `coronablink` provides **primitives** rather than black-box functions. You compose the analysis pipeline that fits your science case.

## Documentation

Full documentation is available at [coronablink.readthedocs.io](https://coronablink.readthedocs.io).
