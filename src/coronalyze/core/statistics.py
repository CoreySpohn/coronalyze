"""Statistical functions for masked arrays and small-sample corrections.

Implements JAX-native masked statistics and the Mawet et al. (2014)
small-sample penalty for high-contrast imaging SNR calculations.
"""

import jax.numpy as jnp


def masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> float:
    """Compute the mean of masked values.

    Args:
        values: 1D array of values.
        mask: Boolean mask (True for valid values).

    Returns:
        Mean of valid values, or 0 if no valid values.
    """
    count = jnp.sum(mask)
    masked_sum = jnp.sum(values * mask)
    return masked_sum / jnp.maximum(count, 1.0)


def masked_std(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    mean: float | None = None,
) -> float:
    """Compute the standard deviation of masked values.

    Uses Bessel's correction (N-1 denominator) for unbiased estimation.

    Args:
        values: 1D array of values.
        mask: Boolean mask (True for valid values).
        mean: Pre-computed mean. If None, computed from masked values.

    Returns:
        Standard deviation of valid values.
    """
    if mean is None:
        mean = masked_mean(values, mask)

    count = jnp.sum(mask)
    residuals = (values - mean) * mask
    variance = jnp.sum(residuals**2) / jnp.maximum(count - 1, 1.0)
    return jnp.sqrt(variance)


def small_sample_penalty(n: int | jnp.ndarray) -> float:
    """Compute the Mawet et al. (2014) small-sample statistics correction.

    At small angular separations, fewer reference apertures are available,
    which inflates the noise estimate. This penalty factor accounts for
    the additional uncertainty.

    Reference: Mawet et al. (2014) ApJ
               Equation 9: sigma_corrected = sigma * sqrt(1 + 1/n)

    Args:
        n: Number of reference apertures.

    Returns:
        Correction factor sqrt(1 + 1/n).
    """
    return jnp.sqrt(1 + 1 / jnp.maximum(n, 1.0))
