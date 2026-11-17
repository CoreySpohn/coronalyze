"""Analysis subpackage for coronalyze.

Contains high-level analysis workflows including yield estimation tools.
"""

from coronalyze.analysis.yields import (
    get_perfect_residuals,
)

__all__ = [
    "get_perfect_residuals",
]
