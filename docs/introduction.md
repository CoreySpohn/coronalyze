# Introduction to coronalyze

coronalyze is the analysis companion to coronagraphoto. It handles post-processing—everything that happens after the photons hit the detector: PSF subtraction, signal extraction, and SNR estimation.

This document covers the conceptual foundation, implementation details, and practical usage.

---

## Motivation

The HWO simulation ecosystem separates **image generation** from **image analysis**:

- **coronagraphoto** simulates the physics: light propagation, coronagraph PSFs (via yippy), detector noise, zodiacal backgrounds.
- **coronalyze** does the math: PCA subtraction, aperture photometry, detection statistics.

This boundary keeps the simulation agnostic to downstream analysis choices, and lets the analysis code run on any image—synthetic or real.

---

## Core Ideas

### JAX-Native Design

Everything in coronalyze runs through JAX. That means:

1. **JIT Compilation**: Functions compile to XLA and run on CPU or GPU with no code changes.
2. **Differentiability**: You can backpropagate through the entire pipeline. This enables gradient-based optimization of coronagraph masks, observer integration times, or orbital parameters.
3. **Vectorization**: `vmap` replaces Python loops. Checking 10,000 candidate positions takes roughly the same time as checking one.

### SNR Estimation

coronalyze implements the **Mawet et al. (2014)** method for signal-to-noise estimation:

- Places discrete apertures around the target separation
- Applies Small-Sample Statistics corrections
- Matches the methodology used in VIP and pyKLIP

Use it when you need to publish a detection significance or compare with community libraries.

---

## Implementation

### Package Layout

```
coronalyze/
├── core/          # Low-level JAX primitives
│   ├── geometry.py      # Radial distances, aperture coordinates
│   ├── photometry.py    # Convolution kernels, flux maps
│   ├── snr.py           # Mawet SNR, batch versions
│   ├── matched_filter.py # Experimental matched-filter SNR
│   ├── pca.py           # Snapshot KLIP
│   └── statistics.py    # Masked mean, std, small-sample correction
├── pipelines/     # High-level workflows
│   └── yield_pipeline.py  # klip_subtract, calculate_yield_snr
└── interfaces/    # Adapters for other packages
    └── coronagraphoto.py
```

The `core/` modules are stateless: pure functions, no classes, no side effects. Everything in `core/` is designed to be traced and JIT-compiled.

### Handling Static Shapes

JAX requires array shapes to be known at trace time. This conflicts with astronomy code where aperture size depends on runtime parameters like FWHM.

The pattern we use:

1. A **Python wrapper** computes concrete integers (kernel size, aperture count) from the float FWHM.
2. It builds the necessary arrays (convolution kernels, coordinate buffers).
3. A **JIT-compiled inner function** receives those arrays and performs the actual math.

This keeps the public API flexible while satisfying JAX's constraints.

### Batched Extraction

The naive approach—looping over planet candidates—wastes work because the expensive image convolution gets repeated. Instead:

1. Compute the flux map (convolution with aperture kernel) **once**.
2. Use `jax.vmap` to extract values at many (y, x) positions in parallel.

The `snr()` function implements this pattern. 100 candidates run in roughly the time of 1.

---

## Usage

### Basic SNR Calculation

```python
from coronalyze import snr
import jax.numpy as jnp

# Your post-processed residual image
residual = ...  # shape (H, W)

# Planet locations in (y, x) pixel coordinates
positions = jnp.array([[100.5, 120.3]])  # (N, 2) array

# Resolution element size in pixels
fwhm = 3.5

snr_values = snr(residual, positions, fwhm)
```

### Yield Pipelines

For mission simulations, use the subtraction primitives:

```python
from coronalyze import subtract_star, subtract_disk, snr
import jax.numpy as jnp

# From coronagraphoto
science = noisy_observation      # counts
star_model = noiseless_star      # counts (expected)
disk_model = noiseless_disk      # counts (expected, or zeros)

# Subtract stellar PSF
residual = subtract_star(science, star_model)

# Optionally subtract disk model
residual = subtract_disk(residual, disk_model)

# Batch SNR for multiple candidates
positions = jnp.array([[100.5, 120.3], [100.5, 80.0]])
snr_values = snr(residual, positions, fwhm=3.5)
```

### High-Performance Pipelines

For iterative pipelines, use the `snr_estimator()` factory:

```python
from coronalyze import snr_estimator
import jax

# Pre-compute aperture kernel once
estimator = snr_estimator(fwhm=3.5, fast=True)

# Use in JIT-compiled loop
@jax.jit
def process_cube(images, positions):
    return jax.vmap(lambda img: estimator(img, positions))(images)
```

### Integration with coronagraphoto

coronalyze takes plain JAX arrays. It doesn't require coronagraphoto objects:

```python
# If you have a DetectorReadout object:
image = readout.rate  # or readout.readout for counts

# Pass directly to analysis:
snr_values = snr(image, positions, fwhm)
```

This means you can use coronalyze on testbed data, archival observations, or outputs from other simulators—anything that gives you a 2D array.

---

## Further Reading

- `docs/examples/snr_methods.ipynb` – SNR calculation demonstration
- `tests/test_snr_physics.py` – Physics-based validation tests
- `docs/examples/coronagraphoto_pipeline.ipynb` – End-to-end example with coronagraphoto
