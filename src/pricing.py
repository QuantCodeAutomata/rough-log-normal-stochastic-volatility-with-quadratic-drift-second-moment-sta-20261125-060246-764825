"""Black's formula and conditional Monte Carlo option pricing for the
log-normal Volterra model, used to calibrate the model to option smiles
(paper Appendix B.3, Eq. B6; Sec. 5).

scipy.stats.norm.cdf is used for the standard normal CDF; the conditional
Monte Carlo estimator itself is a custom implementation of Eq. (B6), since no
general-purpose library implements this model-specific conditional pricing
formula.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.markovian_kernel import KernelSpec
from src.simulation import ModelParams, simulate_paths


def black_price(forward, strike, total_var, option_type: str = "call"):
    """Black (1976) undiscounted forward price of a European option, with
    ``total_var`` = sigma^2 * T the total variance (Appendix B.3).

    Vectorised over ``forward``/``total_var`` (e.g. one value per simulated
    path); ``strike`` and ``option_type`` are scalar per call. Degenerates to
    the intrinsic value at total_var=0, consistent with the Black formula's
    zero-variance limit.
    """
    forward = np.asarray(forward, dtype=float)
    total_var = np.asarray(total_var, dtype=float)
    if np.any(total_var < 0):
        raise ValueError("total_var must be non-negative")
    intrinsic = (np.maximum(forward - strike, 0.0) if option_type == "call"
                 else np.maximum(strike - forward, 0.0))
    safe_var = np.where(total_var > 0.0, total_var, 1.0)
    vol_sqrt_t = np.sqrt(safe_var)
    d1 = (np.log(forward / strike) + 0.5 * safe_var) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if option_type == "call":
        price = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    elif option_type == "put":
        price = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return np.where(total_var > 0.0, price, intrinsic)


def implied_vol_from_price(forward: float, strike: float, T: float, price: float,
                            option_type: str = "call") -> float:
    """Invert Black's formula for the implied volatility via bisection on total
    variance; used only to express model/market prices on a common IV axis
    (Sec. 5, Figure 1)."""
    intrinsic = max(forward - strike, 0.0) if option_type == "call" else max(strike - forward, 0.0)
    if price <= intrinsic:
        return 0.0
    lo, hi = 1e-8, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if black_price(forward, strike, mid * mid * T, option_type) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def conditional_mc_price(
    kernel: KernelSpec,
    params: ModelParams,
    T: float,
    strikes: np.ndarray,
    option_types: list[str],
    steps_per_year: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Conditional Monte Carlo price of a vector of strikes at common forward 1
    and maturity T (Appendix B.3, Eq. B6): conditional on the Brownian path of
    the common factor sigma^N, the log-price is Gaussian with forward
    F_n = exp(rho*J_n - 0.5*rho^2*I_n) and total variance v_n=(1-rho^2)*I_n, so
    the undiscounted price is the path-average of Black's formula.

    This implements the conditional-MC formula of Eq. (B6) directly (without
    the Gaussian half-tilt variance reduction of Eq. B7-B8, which is a
    variance-reduction device for the paper's real-data calibration and is not
    exercised here; see results/RESULTS.md).
    """
    out = simulate_paths(kernel, params, T=T, steps_per_year=steps_per_year,
                          n_paths=n_paths, seed=seed)
    rho = out["rho"]
    F_n = np.exp(rho * out["J"] - 0.5 * rho ** 2 * out["I"])
    v_n = (1.0 - rho ** 2) * out["I"]
    prices = np.empty(len(strikes))
    for i, (K, opt) in enumerate(zip(strikes, option_types)):
        prices[i] = black_price(F_n, K, v_n, opt).mean()
    return prices
