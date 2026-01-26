"""Interface adapters for integrating with external packages."""

from coronablink.interfaces.coronagraphoto import (
    analyze_observation,
    extract_image,
    get_fwhm,
)

__all__ = [
    "analyze_observation",
    "extract_image",
    "get_fwhm",
]
