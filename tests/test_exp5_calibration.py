"""Fast unit tests for the Experiment 5 calibration machinery (not the full,
slow synthetic self-test run by ``python -m exp.exp_5``).
"""

from __future__ import annotations

import numpy as np

from src.calibration import SmileQuote, calibrate, price_quotes
from src.config import EXP5_KAPPA2_MARGIN, EXP5_TABLE4_REFERENCE, KERNEL_1F_NODES, KERNEL_1F_WEIGHTS
from src.markovian_kernel import KernelSpec

_KERNEL_1F = KernelSpec(nodes=KERNEL_1F_NODES, weights=KERNEL_1F_WEIGHTS)
_PARAM_KEYS = ("sigma0", "theta", "kappa1", "kappa2", "beta", "eps")


def test_smile_quote_mid_and_spread() -> None:
    quotes = [
        SmileQuote(strike=0.9, maturity=0.25, option_type="put", bid=0.02, ask=0.03),
        SmileQuote(strike=1.0, maturity=0.25, option_type="call", bid=0.05, ask=0.06),
        SmileQuote(strike=1.1, maturity=0.5, option_type="call", bid=0.01, ask=0.015),
    ]
    assert quotes[0].mid == 0.025
    assert abs(quotes[0].spread - 0.01) < 1e-12
    assert quotes[1].mid == 0.055
    assert abs(quotes[2].spread - 0.005) < 1e-12


def test_price_quotes_finite_and_bounded() -> None:
    ref = EXP5_TABLE4_REFERENCE["BTC_H0.50"]
    params = {k: ref[k] for k in _PARAM_KEYS}
    quotes = [
        SmileQuote(strike=0.8, maturity=0.25, option_type="put", bid=0.0, ask=0.0),
        SmileQuote(strike=1.0, maturity=0.25, option_type="call", bid=0.0, ask=0.0),
        SmileQuote(strike=1.2, maturity=0.25, option_type="call", bid=0.0, ask=0.0),
    ]
    prices = price_quotes(_KERNEL_1F, params, quotes, steps_per_year=20, n_paths=200, seed=1)

    forward = 1.0
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)
    for price, quote in zip(prices, quotes):
        if quote.option_type == "call":
            assert price <= forward + 1e-8


def test_calibrate_from_ground_truth_honours_martingale_constraint() -> None:
    ref = EXP5_TABLE4_REFERENCE["BTC_H0.50"]
    params = {k: ref[k] for k in _PARAM_KEYS}
    x0 = np.array([ref[k] for k in _PARAM_KEYS], dtype=float)

    # steps_per_year=50 keeps N_steps = round(T * steps_per_year) >= 1 for these
    # short maturities (30d, 90d); steps_per_year=20 would round T=7/365 to 0 steps.
    # scipy's 'lm' method (used inside calibrate) requires at least as many residuals
    # as free variables (6 here), hence 3 strikes x 2 maturities = 6 quotes below.
    quotes_raw = [
        SmileQuote(strike=0.8, maturity=30 / 365, option_type="put", bid=0.0, ask=0.0),
        SmileQuote(strike=0.9, maturity=30 / 365, option_type="put", bid=0.0, ask=0.0),
        SmileQuote(strike=1.1, maturity=30 / 365, option_type="call", bid=0.0, ask=0.0),
        SmileQuote(strike=0.9, maturity=90 / 365, option_type="put", bid=0.0, ask=0.0),
        SmileQuote(strike=1.1, maturity=90 / 365, option_type="call", bid=0.0, ask=0.0),
        SmileQuote(strike=1.2, maturity=90 / 365, option_type="call", bid=0.0, ask=0.0),
    ]
    mids = price_quotes(_KERNEL_1F, params, quotes_raw, steps_per_year=50, n_paths=200, seed=3)
    quotes = [
        SmileQuote(strike=q.strike, maturity=q.maturity, option_type=q.option_type,
                    bid=mid * 0.99, ask=mid * 1.01)
        for q, mid in zip(quotes_raw, mids)
    ]

    fitted = calibrate(_KERNEL_1F, quotes, x0, steps_per_year=50, n_paths=200, seed=3,
                        H=0.50, kappa2_margin=EXP5_KAPPA2_MARGIN)

    assert fitted["kappa2"] >= fitted["beta"]
    assert fitted["kappa2"] >= fitted["beta"] + EXP5_KAPPA2_MARGIN - 1e-6
    assert np.isfinite(fitted["weighted_rms"])
    assert fitted["n_quotes"] == len(quotes)
