"""Sella physics constants.

Confirmed against Sella master source (https://github.com/zadorlab/sella,
sella/optimize/optimize.py and sella/peswrapper.py) and Hermes et al.,
JCTC 2022, https://doi.org/10.1021/acs.jctc.2c00395.

These constants describe Sella's trust-region control parameters. v1
dashboard work does not yet consume them; they are named here so the
v2 stuck-opt detector (rtrust-at-floor, rho-outside-band) does not have
to rediscover the values by reading Sella source.

Trust-region ratio thresholds are mode-dependent. A "bad step" in Sella
is rho < 1/rho_dec or rho > rho_dec. The generic "rho < 0.25 = bad"
heuristic from the wider trust-region literature does NOT apply to
Sella - the per-mode rho_dec values below are authoritative.
"""

from __future__ import annotations

RHO_DEC_SADDLE: float = 5.0
"""Trust-region ratio bound for saddle-point mode (order >= 1).

Bad-step condition: rho < 0.2 or rho > 5.0.
"""

RHO_DEC_MINIMUM: float = 100.0
"""Trust-region ratio bound for minimum mode (order == 0).

Bad-step condition: rho < 0.01 or rho > 100.
"""

TRUST_RADIUS_FLOOR_ETA: float = 1e-4
"""Minimum value of ``self.delta`` (the trust radius).

Sella clamps the trust radius at this floor and cannot shrink further.
A run that sits at this floor with non-zero fmax is the canonical
"stuck" pathology.
"""

TRUST_RADIUS_INITIAL_DELTA0: float = 0.1
"""Starting value of ``self.delta`` on ``Sella.__init__``.

Reset to this value on every restart because the trust radius is not
persisted in the trajectory.
"""


def rho_dec_for_order(order: int) -> float:
    """Return the mode-appropriate rho_dec bound.

    Args:
        order: Sella saddle order (0 = minimum, >= 1 = saddle/TS).

    Returns:
        RHO_DEC_MINIMUM for order == 0, RHO_DEC_SADDLE otherwise.
    """
    return RHO_DEC_MINIMUM if order == 0 else RHO_DEC_SADDLE
