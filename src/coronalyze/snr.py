"""Signal-to-noise ratio calculations for planet detection."""

import jax.numpy as jnp


def calculate_snr(
    signal: float,
    background_noise: float,
    read_noise: float = 0.0,
    dark_current: float = 0.0,
) -> float:
    """Calculate signal-to-noise ratio for a detection.

    Uses the CCD equation for SNR:
        SNR = S / sqrt(S + B + R^2 + D)

    where:
        S = signal (electrons)
        B = background noise (electrons)
        R = read noise (electrons)
        D = dark current (electrons)

    Args:
        signal: Source signal in electrons.
        background_noise: Background noise in electrons (from sky, zodi, etc.).
        read_noise: Read noise in electrons (per pixel, summed over aperture).
        dark_current: Dark current in electrons (summed over aperture).

    Returns:
        Signal-to-noise ratio.
    """
    variance = signal + background_noise + read_noise**2 + dark_current
    return signal / jnp.sqrt(variance)


def exposure_time_for_snr(
    target_snr: float,
    signal_rate: float,
    background_rate: float,
    read_noise: float = 0.0,
    dark_rate: float = 0.0,
) -> float:
    """Calculate exposure time needed to achieve a target SNR.

    Solves the CCD equation for exposure time:
        SNR = S*t / sqrt(S*t + B*t + R^2 + D*t)

    This is a quadratic equation in t.

    Args:
        target_snr: Desired signal-to-noise ratio.
        signal_rate: Source signal rate in electrons/second.
        background_rate: Background rate in electrons/second.
        read_noise: Read noise in electrons (constant, not per second).
        dark_rate: Dark current rate in electrons/second.

    Returns:
        Required exposure time in seconds.
    """
    # SNR^2 * (S*t + B*t + R^2 + D*t) = S^2 * t^2
    # SNR^2 * (S + B + D) * t + SNR^2 * R^2 = S^2 * t^2
    # S^2 * t^2 - SNR^2 * (S + B + D) * t - SNR^2 * R^2 = 0

    a = signal_rate**2
    b = -(target_snr**2) * (signal_rate + background_rate + dark_rate)
    c = -(target_snr**2) * read_noise**2

    # Quadratic formula (take positive root)
    discriminant = b**2 - 4 * a * c
    t = (-b + jnp.sqrt(discriminant)) / (2 * a)

    return t
