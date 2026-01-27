<p align="center">
  <a href="https://pypi.org/project/coronalyze/"><img src="https://img.shields.io/pypi/v/coronalyze.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://coronalyze.readthedocs.io"><img src="https://readthedocs.org/projects/coronalyze/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/coronalyze/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/coronalyze/tests.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/coronalyze/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/CoreySpohn/coronalyze"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg?style=flat-square&logo=python" alt="Python Versions"></a>
</p>

---

# coronalyze

**coronalyze** is a JAX-based Python library for post-processing coronagraphic direct imaging data. It is designed as a companion to **[coronagraphoto](https://github.com/CoreySpohn/coronagraphoto)**, providing analysis tools for simulated and (maybe) real coronagraphic observations.

## Key Features

*   **SNR Estimation**: Mawet et al. (2014) aperture photometry SNR with small-sample corrections
*   **PSF Subtraction**: KLIP/PCA and reference differential imaging
*   **Planet Injection**: Forward modeling for yield simulations
*   **JAX-Native**: Fully JIT-compilable, differentiable, and GPU-accelerated

## Installation

```bash
pip install coronalyze
```

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements.)*

## Quick Start

```python
import coronalyze as cz
import jax.numpy as jnp

# Calculate SNR at planet positions using Mawet et al. (2014)
snr_values = cz.snr(residual_image, planet_positions, fwhm=4.5)

# Or use the estimator pattern for efficient batch processing
estimator = cz.snr_estimator(fwhm=4.5)
snr_values = estimator(residual_image, planet_positions)
```

## Pairing with coronagraphoto

```python
import coronagraphoto as cg
import coronalyze as cz

# Generate image with coronagraphoto
obs = cg.Observation(scene, coronagraph, settings)
obs.create_images()
science_image = obs.get_total_image()

# Get the star model for subtraction
star_model = obs.get_star_image()

# Subtract and calculate SNR
residual = cz.subtract_star(science_image, star_model)
snr_values = cz.snr(residual, planet_positions, fwhm=4.5)
```

## Design Philosophy

Like its companion `coronagraphoto`, `coronalyze` provides **primitives** rather than black-box functions. You compose the analysis pipeline that fits your science case.

## Documentation

Full documentation is available at [coronalyze.readthedocs.io](https://coronalyze.readthedocs.io).

