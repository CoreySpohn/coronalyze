"""coronalyze: JAX-based post-processing for coronagraphic direct imaging.

This library provides analysis tools for coronagraphic observations,
designed as a companion to coronagraphoto.
"""

from coronalyze.photometry import aperture_photometry, circular_aperture_mask
from coronalyze.snr import calculate_snr

__all__ = [
    "aperture_photometry",
    "calculate_snr",
    "circular_aperture_mask",
]
