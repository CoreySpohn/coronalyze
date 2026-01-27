"""Simplified analysis tools for yield modeling and noise floor estimation.

Implements 'Faux-RDI' and related tools for yield calculations where we assume
perfect knowledge of the background model (stellar speckles + disk).
"""

import jax
import jax.numpy as jnp


def get_perfect_residuals(
    observation: jnp.ndarray,
    expectation_model: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate residuals assuming perfect subtraction of static structure.

    This simulates 'Faux-RDI': we subtract the exact expectation value of
    the star and disk. The residual contains only the fundamental noise
    (photon + read noise) and any signals not in the model (planets).

    Args:
        observation: The noisy data (Data = Poisson(Model) + ReadNoise).
        expectation_model: The noiseless expectation of background sources
            (stellar PSF + disk, excluding planets).

    Returns:
        Residual image containing noise + unmodeled signals (planets).
    """
    return observation - expectation_model


def get_photon_noise_map(
    expectation_rate: jnp.ndarray,
    exposure_time: float,
    read_noise: float = 0.0,
) -> jnp.ndarray:
    """Calculate the theoretical 1-sigma noise map in rate units.

    Properly converts between rate and count units to combine photon noise
    with read noise. Returns noise in the same units as input (counts/sec).

    Formula: Sigma_rate = sqrt(Rate * t + RN^2) / t

    Args:
        expectation_rate: Expected count rate image (counts/sec).
        exposure_time: Integration time in seconds.
        read_noise: Read noise in electrons (per pixel). Default 0.

    Returns:
        1-sigma noise map in rate units (counts/sec), same shape as input.
    """
    # Convert rate to counts for proper Poisson statistics
    total_counts = jnp.maximum(expectation_rate, 0.0) * exposure_time
    # Variance in counts: Poisson variance + read noise^2
    variance_counts = total_counts + read_noise**2
    # Convert back to rate units
    sigma_counts = jnp.sqrt(variance_counts)
    return sigma_counts / exposure_time


def simulate_observation(
    clean_signal: jnp.ndarray,
    background_model: jnp.ndarray,
    exposure_time: float = 1.0,
    rng_key: jax.Array = None,
) -> jnp.ndarray:
    """Generate a noisy realization of the scene for yield tests.

    Applies Poisson noise to the combined scene (signal + background).
    Returns data in rate units (counts/sec).

    Args:
        clean_signal: Planet image (counts/sec).
        background_model: Star + Disk expectation (counts/sec).
        exposure_time: Integration time (seconds).
        rng_key: JAX PRNG Key. If None, uses key 0.

    Returns:
        Noisy image in units of counts/sec (Poisson noise added in counts,
        then divided back by exposure time).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Total rate
    total_flux = (clean_signal + background_model) * exposure_time

    # Poisson draw (expects non-negative rates)
    counts = jax.random.poisson(rng_key, jnp.maximum(total_flux, 0.0))

    # Return to rate units
    return counts / exposure_time
