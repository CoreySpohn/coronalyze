"""Analysis subpackage for coronablink.

Contains high-level analysis workflows including yield estimation tools.
"""

from coronablink.analysis.yields import (
    get_perfect_residuals,
    get_photon_noise_map,
    simulate_observation,
)

__all__ = [
    "get_perfect_residuals",
    "get_photon_noise_map",
    "simulate_observation",
]
