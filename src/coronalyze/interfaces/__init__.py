"""Interface adapters for integrating with external packages."""

from coronalyze.interfaces.coronagraphoto import (
    analyze_observation,
    extract_image,
    get_fwhm,
    sky_offset_to_position,
)

__all__ = [
    "analyze_observation",
    "extract_image",
    "get_fwhm",
    "sky_offset_to_position",
]
