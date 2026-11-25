"""Fractional kernel, resolvent, second-moment stability parameter, conditional
moments and the exponential growth rate for the Riemann-Liouville kernel
K(t) = t^{H-1/2}/Gamma(H+1/2) with linear drift (Sec. 3.1-3.3 of the paper).

Custom implementation of the paper's own formulas (no general-purpose library
implements this model); scipy is used only for standard numerical primitives
(quadrature, root finding, interpolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, interpolate, optimize
from scipy.special import gamma as gamma_fn

from src.mittag_leffler import mittag_leffler


def alpha_of_H(H: float) -> float:
    """alpha = H + 1/2, the Mittag-Leffler order of the Riemann-Liouville kernel."""
    return H + 0.5


def C_alpha(alpha: float) -> float:
    """Spectral constant C(alpha) in the Parseval formula for int_0^inf G(u)^2 du
    (paper Sec. 3.1, 3.3): C(alpha) = -cot(pi alpha/2)/(alpha sin(pi/alpha)) for
    1/2 < alpha < 1, and C(1) = 1/2 by the alpha -> 1 (H -> 1/2) limit.
    """
    if np.isclose(alpha, 1.0, atol=1e-10):
        return 0.5
    return -1.0 / np.tan(np.pi * alpha / 2.0) / (alpha * np.sin(np.pi / alpha))


def integral_G_squared_quad(alpha: float, kappa1: float) -> float:
    """Independent cross-check of int_0^inf G(u)^2 du via direct spectral
    (Parseval) quadrature: (1/pi) int_0^inf domega / (omega^{2a}+2k cos(pi a/2) omega^a+k^2).
    """
    if kappa1 == 0.0:
        return np.inf

    def integrand(omega):
        return 1.0 / (
            omega ** (2 * alpha)
            + 2 * kappa1 * np.cos(np.pi * alpha / 2.0) * omega ** alpha
            + kappa1 ** 2
        )

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    return value / np.pi


def integral_G_squared(alpha: float, kappa1: float) -> float:
    """Closed-form int_0^inf G(u)^2 du = C(alpha) * kappa1^{-2H/alpha} (Sec. 3.3)."""
    if kappa1 == 0.0:
        return np.inf
    H = alpha - 0.5
    return C_alpha(alpha) * kappa1 ** (-2.0 * H / alpha)


def theta_c(alpha: float, kappa1: float) -> float:
    """Critical volatility-of-volatility theta_c = (int_0^inf G^2)^{-1/2} (Eq. 3.3)."""
    return 1.0 / np.sqrt(integral_G_squared(alpha, kappa1))


def gamma_param(theta: float, alpha: float, kappa1: float) -> float:
    """Stability parameter gamma = theta^2 int_0^inf G(u)^2 du (Sec. 3.3)."""
    return theta ** 2 * integral_G_squared(alpha, kappa1)


def innovation_proxy(theta: float, H: float, Delta: float) -> float:
    """One-step innovation proxy nu_Delta = theta Delta^H / (sqrt(2H) Gamma(alpha)) (Sec. 3.3)."""
    alpha = alpha_of_H(H)
    return theta * Delta ** H / (np.sqrt(2.0 * H) * gamma_fn(alpha))


@dataclass
class ResolventTable:
    """Interpolated E_{alpha,alpha}(-kappa1 t^alpha) and E_{alpha,1}(-kappa1 t^alpha)
    on [0, t_max], built from a modest number of exact (mpmath-based) Mittag-Leffler
    evaluations and used to evaluate the resolvent G(t) and the conditional mean
    cheaply and densely on the product-integration grid (avoids calling the
    arbitrary-precision Mittag-Leffler series millions of times).
    """

    alpha: float
    kappa1: float
    t_max: float
    _Ghat: interpolate.PchipInterpolator
    _Ehat: interpolate.PchipInterpolator

    @classmethod
    def build(cls, alpha: float, kappa1: float, t_max: float, n_points: int = 250) -> "ResolventTable":
        # Grid concentrated near zero (log-spaced) where the Mittag-Leffler argument
        # kappa1*t^alpha changes fastest, plus a uniform tail.
        t_small = np.geomspace(max(t_max * 1e-6, 1e-8), max(t_max * 0.1, 1e-6), n_points // 2)
        t_large = np.linspace(t_small[-1], t_max, n_points - len(t_small) + 1)[1:]
        t_grid = np.concatenate(([0.0], t_small, t_large))
        z = -kappa1 * t_grid ** alpha
        Ghat_vals = mittag_leffler(z, alpha, alpha)  # E_{alpha,alpha}(-kappa1 t^alpha)
        Ehat_vals = mittag_leffler(z, alpha, 1.0)  # E_{alpha,1}(-kappa1 t^alpha)
        return cls(
            alpha=alpha, kappa1=kappa1, t_max=t_max,
            _Ghat=interpolate.PchipInterpolator(t_grid, Ghat_vals),
            _Ehat=interpolate.PchipInterpolator(t_grid, Ehat_vals),
        )

    def G(self, t: np.ndarray) -> np.ndarray:
        """Resolvent-based kernel G(t) = t^{alpha-1} E_{alpha,alpha}(-kappa1 t^alpha)."""
        t = np.asarray(t, dtype=float)
        out = np.empty_like(t)
        zero = t <= 0.0
        out[~zero] = t[~zero] ** (self.alpha - 1.0) * self._Ghat(t[~zero])
        out[zero] = np.inf if self.alpha < 1.0 else self._Ghat(0.0)
        return out

    def E_alpha(self, t: np.ndarray) -> np.ndarray:
        """E_alpha(-kappa1 t^alpha) = E_{alpha,1}(-kappa1 t^alpha), used by the
        conditional mean m(t) = theta + (sigma0-theta) E_alpha(-kappa1 t^alpha)."""
        return self._Ehat(np.asarray(t, dtype=float))


def conditional_mean(t: np.ndarray, table: ResolventTable, sigma0: float, theta: float) -> np.ndarray:
    """m(t) = theta + (sigma0 - theta) E_alpha(-kappa1 t^alpha), starting from t=0 (Sec. 3.2)."""
    return theta + (sigma0 - theta) * table.E_alpha(t)


def conditional_second_moment(
    table: ResolventTable,
    sigma0: float,
    theta: float,
    T: float,
    steps_per_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the linear Volterra equation v(s) = m(s)^2 + theta^2 int_0^s G(s-r)^2 v(r) dr
    (Sec. 3.2, Eq. (14) at t=0) by explicit product integration on a uniform grid.

    The kernel K2(u) = theta^2 G(u)^2 ~ theta^2 u^{2(alpha-1)}/Gamma(alpha)^2 as u->0 is
    weakly singular but integrable (2(alpha-1) > -1 for alpha > 1/2). Each subinterval
    [(m-1)h, m h] is integrated exactly against the leading power-law singularity using a
    local linear Taylor expansion of Ghat(u)^2 = E_{alpha,alpha}(-kappa1 u^alpha)^2 around
    u=0 for m=1, and plain trapezoidal quadrature on the (regular) kernel for m>=2.
    """
    N = int(round(T * steps_per_year))
    h = T / N
    t_grid = np.arange(N + 1) * h

    alpha = table.alpha
    p = 2.0 * (alpha - 1.0)  # power of the leading singularity of G(u)^2

    m_grid = conditional_mean(t_grid, table, sigma0, theta)
    v = np.empty(N + 1)
    v[0] = sigma0 ** 2

    # Phi[m] = int_{(m-1)h}^{m h} K2(u) du for m = 1, ..., N (lag in grid steps).
    Phi = np.empty(N + 1)  # Phi[0] unused
    Ghat2 = lambda u: table._Ghat(u) ** 2  # noqa: E731 (local helper, smooth for u>0)

    # m = 1: singular subinterval, local expansion Ghat(u)^2 ~ g0 + g1 * u + O(u^2).
    du = 1e-6 * h
    g0 = Ghat2(np.array([du]))[0]
    g1 = (Ghat2(np.array([2 * du]))[0] - g0) / du
    Phi[1] = theta ** 2 * (g0 * h ** (p + 1) / (p + 1) + g1 * h ** (p + 2) / (p + 2))

    # m >= 2: regular kernel, trapezoidal rule using the interpolated Ghat.
    if N >= 1:
        u_nodes = np.arange(1, N + 1) * h
        K2_nodes = theta ** 2 * u_nodes ** p * Ghat2(u_nodes)
        Phi[2:] = 0.5 * h * (K2_nodes[:-1] + K2_nodes[1:])

    for n in range(1, N + 1):
        lag_weights = Phi[n:0:-1]  # Phi[n], Phi[n-1], ..., Phi[1]
        v[n] = m_grid[n] ** 2 + float(np.dot(v[:n], lag_weights))

    return t_grid, m_grid, v


def h_hat_integral(lam: float, theta: float, alpha: float, kappa1: float) -> float:
    """h_hat(lambda) = theta^2/pi int_0^inf domega / |(lambda/2 + i omega)^alpha + kappa1|^2
    (Sec. 3.3), whose root above gamma>1 gives the exponential growth rate lambda*.
    """

    def integrand(omega):
        z = complex(lam / 2.0, omega) ** alpha + kappa1
        return 1.0 / (abs(z) ** 2)

    value, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    return theta ** 2 / np.pi * value


def growth_rate_lambda_star(theta: float, alpha: float, kappa1: float,
                             lam_hi: float = 1.0e6) -> float:
    """Solve h_hat(lambda*) = 1 by bisection (Sec. 3.3); requires gamma = theta^2 int G^2 > 1."""
    gamma = gamma_param(theta, alpha, kappa1)
    if gamma <= 1.0:
        raise ValueError("growth_rate_lambda_star requires gamma > 1 (above the stability threshold)")

    def f(lam):
        return h_hat_integral(lam, theta, alpha, kappa1) - 1.0

    lo, hi = 1e-8, lam_hi
    while f(hi) > 0 and hi < 1e12:
        hi *= 10.0
    return optimize.brentq(f, lo, hi, xtol=1e-6, rtol=1e-10)


def mittag_leffler_growth_rate(H: float, theta: float) -> float:
    """a = theta^2 Gamma(2H) / Gamma(H+1/2)^2 in the Mittag-Leffler growth formula
    E[sigma_t^2] = sigma_0^2 E_{2H}(a t^{2H}) for kappa1=kappa2=0 (Corollary 3.3)."""
    alpha = alpha_of_H(H)
    return theta ** 2 * gamma_fn(2.0 * H) / gamma_fn(alpha) ** 2


def mittag_leffler_growth_curve(t: np.ndarray, H: float, sigma0: float, theta: float) -> np.ndarray:
    """E[sigma_t^2] = sigma_0^2 E_{2H}(a t^{2H}) for the linear-drift-free case (Corollary 3.3)."""
    a = mittag_leffler_growth_rate(H, theta)
    z = a * np.asarray(t, dtype=float) ** (2.0 * H)
    return sigma0 ** 2 * mittag_leffler(z, 2.0 * H, 1.0)
