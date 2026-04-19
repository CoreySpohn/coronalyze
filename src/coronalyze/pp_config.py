"""Post-processing configuration for coronagraphic observations.

PPConfig bundles the post-processing knobs shared between the analytic
(jaxedith) and image-level (coronagraphoto) branches of the hwo-dev
ecosystem. jaxedith accesses these fields via duck typing; it does not
import coronalyze.

Follows equinox-best-practices Style A: pure field declarations with
defaults, no custom __init__.
"""

import equinox as eqx


class PPConfig(eqx.Module):
    """Post-processing configuration.

    Args:
        ppfact: Post-processing factor applied to residual speckle
            (EXOSIMS speckle-residual pipeline).
        n_rolls: Number of telescope rolls in the observing sequence.
        ez_ppf: Exozodi post-processing factor (suppression from ADI
            or reference differential imaging applied to exozodi noise
            floor).
    """

    ppfact: float = 1.0
    n_rolls: int = 1
    ez_ppf: float = 1.0
