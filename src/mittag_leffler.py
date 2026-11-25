"""Two-parameter Mittag-Leffler function E_{a,b}(z) = sum_{k>=0} z^k / Gamma(a k + b).

Custom implementation (paper Sec. 3.1, Eq. (12), and Corollary 3.3): no installable
library implements this function in the execution environment (the Rust-backed
``mittagleffler``/``pymittagleffler`` packages identified as covering this
functionality are not available on the configured package index), so the function
is implemented from scratch following the paper's own implementation note: direct
series summation with careful handling of large arguments via the asymptotic
expansion (Podlubny, "Fractional Differential Equations", 1999, Eq. (1.8.17)-(1.8.28)).

Direct term-by-term summation in double precision suffers catastrophic cancellation
for negative real arguments of moderate-to-large magnitude, because the series is
strongly alternating. We use ``mpmath`` arbitrary-precision arithmetic for the
series branch, which removes the cancellation problem (only the final rounding to
a double loses precision), and the standard asymptotic expansion for very large
|z| where the series would need an impractical number of terms.
"""

from __future__ import annotations

import numpy as np
from mpmath import mp, mpf, gamma as mp_gamma, rgamma as mp_rgamma, exp as mp_exp, nsum

_SERIES_DPS = 50  # decimal digits of working precision for the series branch
_MAX_SERIES_TERMS = 600
_LARGE_Z_THRESHOLD = 40.0  # |z| above which the asymptotic expansion is used
_ASYMPTOTIC_TERMS = 12


def _series_value(z: float, a: float, b: float) -> float:
    """E_{a,b}(z) via direct series summation at high working precision."""
    with mp.workdps(_SERIES_DPS):
        zz = mpf(z)

        def term(k):
            return zz ** k * mp_rgamma(a * k + b)

        total = nsum(term, [0, _MAX_SERIES_TERMS])
        return float(total)


def _asymptotic_value(z: float, a: float, b: float) -> float:
    """Asymptotic expansion of E_{a,b}(z) for |z| large (Podlubny 1999, Sec. 1.8).

    For z -> -infinity (real axis, 0 < a < 2) only the algebraic tail survives:
        E_{a,b}(z) ~ - sum_{k=1}^{p} z^{-k} / Gamma(b - a k).
    For z -> +infinity the exponential term dominates:
        E_{a,b}(z) ~ (1/a) z^{(1-b)/a} exp(z^{1/a}) - sum_{k=1}^{p} z^{-k} / Gamma(b - a k).
    """
    with mp.workdps(_SERIES_DPS):
        zz = mpf(z)
        tail = mpf(0)
        for k in range(1, _ASYMPTOTIC_TERMS + 1):
            # 1/Gamma(b - a k) via the reciprocal gamma function, which is entire
            # and vanishes at the non-positive-integer poles of Gamma itself.
            tail -= zz ** (-k) * mp_rgamma(b - a * k)
        if z > 0:
            exponent = zz ** (mpf(1) / a)
            leading = (mpf(1) / a) * zz ** ((1 - b) / a) * mp_exp(exponent)
            total = leading + tail
        else:
            total = tail
        try:
            return float(total)
        except OverflowError:
            return np.inf if z > 0 else 0.0


def mittag_leffler_scalar(z: float, a: float, b: float = 1.0) -> float:
    """Evaluate the two-parameter Mittag-Leffler function at a single real point."""
    if a <= 0:
        raise ValueError("The order a of the Mittag-Leffler function must be positive")
    if z == 0.0:
        return float(mp_rgamma(b))
    if abs(z) <= _LARGE_Z_THRESHOLD:
        return _series_value(z, a, b)
    return _asymptotic_value(z, a, b)


def mittag_leffler(z: np.ndarray | float, a: float, b: float = 1.0) -> np.ndarray | float:
    """Vectorised two-parameter Mittag-Leffler function E_{a,b}(z) for real z.

    Parameters
    ----------
    z : array_like or float
        Real argument(s).
    a, b : float
        Parameters of the Mittag-Leffler function; a > 0.
    """
    z_arr = np.asarray(z, dtype=float)
    scalar_input = z_arr.ndim == 0
    flat = z_arr.reshape(-1)
    out = np.empty_like(flat)
    for i, zi in enumerate(flat):
        out[i] = mittag_leffler_scalar(float(zi), a, b)
    out = out.reshape(z_arr.shape)
    return float(out) if scalar_input else out
