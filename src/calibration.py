"""Least-squares calibration of the six risk-neutral parameters
(sigma0, theta, kappa1, kappa2, beta, eps) to a set of option quotes under
the martingale constraint kappa2 >= beta (paper Sec. 5, Theorem 4.3).

scipy.optimize.least_squares is used for the bounded, constrained nonlinear
least-squares fit; the pricing model itself (conditional Monte Carlo /
Black's formula) is the paper's own custom pricer, not a library component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from src.kernels import alpha_of_H, gamma_param
from src.markovian_kernel import KernelSpec
from src.pricing import conditional_mc_price


@dataclass(frozen=True)
class SmileQuote:
    """One option quote: strike, maturity (years), option type, bid/ask price
    (undiscounted, forward units) and the full bid-ask spread used as the
    error-normalisation unit (Sec. 5)."""

    strike: float
    maturity: float
    option_type: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


# Parameter order used throughout: [sigma0, theta, kappa1, kappa2, beta, eps]
_PARAM_NAMES = ("sigma0", "theta", "kappa1", "kappa2", "beta", "eps")


def _unpack(x: np.ndarray) -> dict:
    return dict(zip(_PARAM_NAMES, x))


def price_quotes(
    kernel: KernelSpec,
    params: dict,
    quotes: list[SmileQuote],
    steps_per_year: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Model prices of every quote, one conditional-MC batch per distinct maturity
    (Appendix B.3): each maturity gets its own simulation to T, since strikes
    sharing a maturity share one common forward."""
    from src.simulation import ModelParams

    model_params = ModelParams(sigma0=params["sigma0"], kappa1=params["kappa1"],
                                kappa2=params["kappa2"], theta=params["theta"],
                                beta=params["beta"], eps=params["eps"])
    maturities = sorted(set(q.maturity for q in quotes))
    prices = np.empty(len(quotes))
    for m_idx, T in enumerate(maturities):
        idx = [i for i, q in enumerate(quotes) if q.maturity == T]
        strikes = np.array([quotes[i].strike for i in idx])
        types = [quotes[i].option_type for i in idx]
        batch_prices = conditional_mc_price(
            kernel, model_params, T, strikes, types,
            steps_per_year=steps_per_year, n_paths=n_paths, seed=seed + m_idx,
        )
        for local_i, global_i in enumerate(idx):
            prices[global_i] = batch_prices[local_i]
    return prices


def calibrate(
    kernel: KernelSpec,
    quotes: list[SmileQuote],
    x0: np.ndarray,
    steps_per_year: int,
    n_paths: int,
    seed: int,
    H: float,
    kappa2_margin: float = 0.01,
) -> dict:
    """Fit (sigma0, theta, kappa1, kappa2, beta, eps) by weighted least squares
    on bid-ask-spread-normalised price errors (Sec. 5): each maturity carries
    equal weight, so errors are additionally divided by the number of quotes
    at that maturity before stacking the residual vector.

    The constraint kappa2 >= beta + kappa2_margin (Theorem 4.3 martingale
    condition) is enforced by optimising over an unconstrained auxiliary
    variable u with kappa2 = beta + kappa2_margin + softplus(u), so every
    trial parameter vector is a true martingale.
    """
    maturities = sorted(set(q.maturity for q in quotes))
    weight_per_quote = np.array([
        1.0 / sum(1 for q in quotes if q.maturity == quote.maturity) for quote in quotes
    ])
    bid = np.array([q.bid for q in quotes])
    ask = np.array([q.ask for q in quotes])
    mid = 0.5 * (bid + ask)
    spread = np.maximum(ask - bid, 1e-8)

    def softplus(u):
        return np.log1p(np.exp(-abs(u))) + max(u, 0.0)

    def unpack_free(z):
        sigma0, theta, kappa1, beta, eps, u = z
        kappa2 = beta + kappa2_margin + softplus(u)
        return dict(sigma0=abs(sigma0), theta=abs(theta), kappa1=abs(kappa1),
                     kappa2=kappa2, beta=beta, eps=abs(eps))

    def residuals(z):
        params = unpack_free(z)
        prices = price_quotes(kernel, params, quotes, steps_per_year, n_paths, seed)
        err = (prices - mid) / spread
        return err * np.sqrt(weight_per_quote * len(maturities))

    beta0 = x0[4]
    kappa2_0 = x0[3]
    u0 = np.log(max(np.exp(max(kappa2_0 - beta0 - kappa2_margin, 1e-6)) - 1.0, 1e-6))
    z0 = np.array([x0[0], x0[1], x0[2], beta0, x0[5], u0])

    result = least_squares(residuals, z0, method="lm", max_nfev=200)
    fitted = unpack_free(result.x)
    prices = price_quotes(kernel, fitted, quotes, steps_per_year, n_paths, seed)
    err = (prices - mid) / spread
    rms = float(np.sqrt(np.mean(err ** 2)))
    inside = int(np.sum((prices >= bid) & (prices <= ask)))

    theta_vol = float(np.sqrt(fitted["beta"] ** 2 + fitted["eps"] ** 2))
    # gamma at kappa2=0, computed with the fractional kernel formula at the fitted
    # (theta, kappa1) and the calibration's Hurst exponent (Sec. 5, "gamma at
    # kappa2=0 using the fractional kernel formula").
    gamma = gamma_param(theta_vol, alpha_of_H(H), fitted["kappa1"])
    return {
        **fitted,
        "rho": fitted["beta"] / theta_vol,
        "kappa2_minus_beta": fitted["kappa2"] - fitted["beta"],
        "gamma": gamma,
        "weighted_rms": rms,
        "inside_bid_ask": inside,
        "n_quotes": len(quotes),
        "model_prices": prices,
        "success": bool(result.success),
    }
