"""Experiment 5: calibration to Bitcoin and Ether option smiles at fixed
Hurst exponent (paper Sec. 5, Table 4, Figure 1).

DATA AVAILABILITY NOTICE
-------------------------
The paper calibrates the model to Deribit Bitcoin option quotes (30 June 2023)
and Ether option quotes (26 October 2021). This replication can only load
market data through the QuantX Dataloader REST API
(https://quantx-api.limex.pro/apis/dataloader) and the Massive REST API;
neither serves historical Deribit crypto-option chains. The paper itself
states that this snapshot is archived and available only on request from the
corresponding author, which is not accessible in this environment.

Consequently, THIS SCRIPT DOES NOT REPRODUCE TABLE 4 OR FIGURE 1 OF THE
PAPER. No "market" quotes are fabricated and no fit to invented data is
presented as a replication of the paper's results. Instead:

1. A SYNTHETIC SELF-TEST of the calibration *machinery* only: the paper's own
   published parameter vectors (Table 4, verbatim in
   ``src.config.EXP5_TABLE4_REFERENCE``) are used as "ground truth" to price a
   synthetic smile with the project's own Monte Carlo pricer, wrapped in a
   synthetic +/-1% bid/ask band; the ground truth is perturbed by about
   +/-20% to build an optimizer starting guess, and ``calibrate`` is checked
   to recover prices consistent with the synthetic quotes while honouring the
   martingale constraint kappa2 >= beta + margin (Theorem 4.3). This
   validates the pricing and least-squares code paths, not the paper's
   reported Bitcoin/Ether fit.
2. The paper's own published Table 4 numbers are printed and stored verbatim,
   labelled "paper-reported values, not recomputed here", for
   ``results/RESULTS.md`` to cite in its "Not reproduced" section.

Outputs (both in ``results/``, no raw/market time series ever written):
    - exp5_metrics.json       : synthetic self-test results + Table 4 reference
    - exp5_synthetic_smile.png: static figure of the H=0.50 self-test smile
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import SmileQuote, calibrate, price_quotes
from src.config import (
    EXP5_BTC_MATURITIES_DAYS,
    EXP5_H_VALUES,
    EXP5_KAPPA2_MARGIN,
    EXP5_SYNTHETIC_N_PATHS,
    EXP5_SYNTHETIC_N_STRIKES,
    EXP5_SYNTHETIC_SEED,
    EXP5_SYNTHETIC_STEPS_PER_YEAR,
    EXP5_TABLE4_REFERENCE,
    KERNEL_1F_NODES,
    KERNEL_1F_WEIGHTS,
    KERNEL_97F_H,
)
from src.markovian_kernel import KernelSpec, build_97_factor_kernel
from src.pricing import black_price, implied_vol_from_price

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_PARAM_KEYS = ("sigma0", "theta", "kappa1", "kappa2", "beta", "eps")  # also the x0 order for calibrate()

PERTURBATION_FRACTION = 0.20  # +/-20% perturbation of ground truth to build the optimizer x0
SYNTHETIC_SPREAD_FRACTION = 0.01  # conservative, explicit synthetic bid/ask half-spread (+/-1% of mid)
MONEYNESS_RANGE = (0.7, 1.3)  # log-spaced strike grid around forward=1.0 (Sec. 5 OTM selection)

# The 97-factor kernel self-test is far more expensive per Monte Carlo path than the
# 1-factor kernel (97x97 memory-map matrix multiply per step vs. a scalar update), so a
# dedicated, reduced path/step budget is used here to keep the whole experiment under a
# few minutes and well under 1.5 GiB of memory, as timed interactively before writing
# this file (price_quotes on 33 quotes: ~0.5s at n_paths=2000, steps_per_year=50, vs.
# ~0.15s for the 1-factor kernel at the full EXP5_SYNTHETIC_* budget of 20000/365).
SELF_TEST_97F_N_PATHS = 2000
SELF_TEST_97F_STEPS_PER_YEAR = 50


def _param_vector(ref: dict) -> np.ndarray:
    """Extract [sigma0, theta, kappa1, kappa2, beta, eps] from a Table 4 reference
    dict, ignoring the derived/reporting keys (rho, kappa2_minus_beta, gamma, rms,
    inside, n_quotes)."""
    return np.array([ref[k] for k in _PARAM_KEYS], dtype=float)


def _perturb_x0(x_true: np.ndarray, rng: np.random.Generator, frac: float) -> np.ndarray:
    """Perturb each of the 6 ground-truth parameters by a uniform +/-frac relative
    shock, to build a non-trivial optimizer starting guess. No manual projection onto
    the martingale constraint is needed: ``calibrate`` reparametrises kappa2 via a
    softplus so every trial vector it evaluates (including this starting guess) is a
    valid martingale, regardless of whether the raw perturbed (kappa2, beta) pair
    happens to violate kappa2 >= beta + margin."""
    shocks = rng.uniform(-frac, frac, size=x_true.shape)
    return x_true * (1.0 + shocks)


def _build_synthetic_quotes(
    kernel: KernelSpec,
    true_params: dict,
    maturities_years: list[float],
    n_strikes: int,
    n_paths: int,
    steps_per_year: int,
    seed: int,
    spread_fraction: float,
) -> list[SmileQuote]:
    """Build a synthetic smile: log-spaced strikes around forward=1.0 for each
    maturity, priced at the ground-truth parameters, wrapped in a +/-spread_fraction
    bid/ask band around the model mid price. Out-of-the-money option types are used
    (put for moneyness<1, call for moneyness>=1), consistent with Sec. 5."""
    moneyness = np.exp(np.linspace(np.log(MONEYNESS_RANGE[0]), np.log(MONEYNESS_RANGE[1]), n_strikes))
    placeholders: list[SmileQuote] = []
    for T in maturities_years:
        for K in moneyness:
            opt_type = "put" if K < 1.0 else "call"
            placeholders.append(SmileQuote(strike=float(K), maturity=T, option_type=opt_type, bid=0.0, ask=0.0))

    mids = price_quotes(kernel, true_params, placeholders, steps_per_year, n_paths, seed)
    quotes = []
    for quote, mid in zip(placeholders, mids):
        half = spread_fraction * mid
        quotes.append(SmileQuote(strike=quote.strike, maturity=quote.maturity,
                                  option_type=quote.option_type, bid=mid - half, ask=mid + half))
    return quotes


def run_self_test(
    label: str,
    kernel: KernelSpec,
    H: float,
    true_ref: dict,
    maturities_days: tuple[int, ...],
    n_strikes: int,
    n_paths: int,
    steps_per_year: int,
    seed: int,
    kappa2_margin: float,
) -> dict:
    """Run one synthetic self-test of the calibration machinery at a fixed Hurst
    exponent, returning a JSON-serialisable metrics dict plus the quotes/fit needed
    for plotting (under the "_plot_data" key, stripped before saving to JSON)."""
    rng = np.random.default_rng(seed)
    true_params = {k: true_ref[k] for k in _PARAM_KEYS}
    maturities_years = [d / 365.0 for d in maturities_days]

    quotes = _build_synthetic_quotes(kernel, true_params, maturities_years, n_strikes,
                                      n_paths, steps_per_year, seed, SYNTHETIC_SPREAD_FRACTION)

    x_true = _param_vector(true_ref)
    x0 = _perturb_x0(x_true, rng, PERTURBATION_FRACTION)

    t_start = time.time()
    fitted = calibrate(kernel, quotes, x0, steps_per_year, n_paths, seed, H, kappa2_margin)
    elapsed = time.time() - t_start

    martingale_ok = bool(fitted["kappa2"] >= fitted["beta"] + kappa2_margin - 1e-6)
    assert martingale_ok, f"{label}: martingale constraint kappa2>=beta+margin violated"
    assert fitted["weighted_rms"] < 5.0, f"{label}: weighted_rms unexpectedly large: {fitted['weighted_rms']}"
    assert fitted["inside_bid_ask"] >= 0

    metrics = {
        "label": label,
        "H": H,
        "n_factors": kernel.n_factors,
        "ground_truth": {k: float(true_ref[k]) for k in _PARAM_KEYS},
        "x0_perturbed": {k: float(v) for k, v in zip(_PARAM_KEYS, x0)},
        "fitted": {k: float(fitted[k]) for k in _PARAM_KEYS},
        "rho": float(fitted["rho"]),
        "kappa2_minus_beta": float(fitted["kappa2_minus_beta"]),
        "gamma": float(fitted["gamma"]),
        "weighted_rms": float(fitted["weighted_rms"]),
        "inside_bid_ask": int(fitted["inside_bid_ask"]),
        "n_quotes": int(fitted["n_quotes"]),
        "martingale_constraint_satisfied": martingale_ok,
        "optimizer_success": bool(fitted["success"]),
        "elapsed_seconds": elapsed,
        "_plot_data": {"quotes": quotes, "fitted": fitted, "maturities_years": maturities_years},
    }
    return metrics


def _plot_h050_self_test(metrics_h050: dict, path: Path) -> None:
    """Static implied-vol plot of the H=0.50 synthetic self-test: bid/ask band vs.
    the recovered fit's implied vols, one panel per maturity."""
    quotes: list[SmileQuote] = metrics_h050["_plot_data"]["quotes"]
    fitted = metrics_h050["_plot_data"]["fitted"]
    maturities_years: list[float] = metrics_h050["_plot_data"]["maturities_years"]
    model_prices = fitted["model_prices"]

    fig, axes = plt.subplots(1, len(maturities_years), figsize=(5 * len(maturities_years), 4), sharey=True)
    if len(maturities_years) == 1:
        axes = [axes]
    forward = 1.0
    for ax, T in zip(axes, maturities_years):
        idx = [i for i, q in enumerate(quotes) if q.maturity == T]
        strikes = np.array([quotes[i].strike for i in idx])
        order = np.argsort(strikes)
        bid_iv, ask_iv, fit_iv = [], [], []
        for i in idx:
            q = quotes[i]
            bid_iv.append(implied_vol_from_price(forward, q.strike, T, q.bid, q.option_type))
            ask_iv.append(implied_vol_from_price(forward, q.strike, T, q.ask, q.option_type))
            fit_iv.append(implied_vol_from_price(forward, q.strike, T, model_prices[i], q.option_type))
        bid_iv, ask_iv, fit_iv = (np.array(a)[order] for a in (bid_iv, ask_iv, fit_iv))
        strikes_sorted = strikes[order]
        ax.fill_between(strikes_sorted, bid_iv, ask_iv, color="tab:blue", alpha=0.3, label="synthetic bid/ask")
        ax.plot(strikes_sorted, fit_iv, "o-", color="tab:red", label="recovered fit")
        ax.set_title(f"T={T * 365:.0f}d")
        ax.set_xlabel("strike (moneyness, forward=1)")
    axes[0].set_ylabel("implied volatility")
    axes[0].legend()
    fig.suptitle("Synthetic self-test (not Figure 1 of the paper) -- H=0.50, BTC-H0.50 reference parameters")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    assert EXP5_H_VALUES == (0.30, 0.50)
    kernel_h050 = KernelSpec(nodes=KERNEL_1F_NODES, weights=KERNEL_1F_WEIGHTS)
    kernel_h030 = build_97_factor_kernel(H=KERNEL_97F_H)

    print("Experiment 5: synthetic self-test of the calibration machinery")
    print("(real Deribit BTC/ETH quotes are not available in this environment; see module docstring)\n")

    t0 = time.time()
    metrics_h050 = run_self_test(
        label="BTC_H0.50_self_test", kernel=kernel_h050, H=0.50,
        true_ref=EXP5_TABLE4_REFERENCE["BTC_H0.50"],
        maturities_days=EXP5_BTC_MATURITIES_DAYS, n_strikes=EXP5_SYNTHETIC_N_STRIKES,
        n_paths=EXP5_SYNTHETIC_N_PATHS, steps_per_year=EXP5_SYNTHETIC_STEPS_PER_YEAR,
        seed=EXP5_SYNTHETIC_SEED, kappa2_margin=EXP5_KAPPA2_MARGIN,
    )
    print(f"[H=0.50, 1-factor kernel] weighted_rms={metrics_h050['weighted_rms']:.4f} "
          f"inside_bid_ask={metrics_h050['inside_bid_ask']}/{metrics_h050['n_quotes']} "
          f"martingale_ok={metrics_h050['martingale_constraint_satisfied']} "
          f"elapsed={metrics_h050['elapsed_seconds']:.1f}s")

    metrics_h030 = run_self_test(
        label="BTC_H0.30_self_test", kernel=kernel_h030, H=0.30,
        true_ref=EXP5_TABLE4_REFERENCE["BTC_H0.30"],
        maturities_days=EXP5_BTC_MATURITIES_DAYS, n_strikes=EXP5_SYNTHETIC_N_STRIKES,
        n_paths=SELF_TEST_97F_N_PATHS, steps_per_year=SELF_TEST_97F_STEPS_PER_YEAR,
        seed=EXP5_SYNTHETIC_SEED, kappa2_margin=EXP5_KAPPA2_MARGIN,
    )
    print(f"[H=0.30, 97-factor kernel, reduced n_paths/steps_per_year] "
          f"weighted_rms={metrics_h030['weighted_rms']:.4f} "
          f"inside_bid_ask={metrics_h030['inside_bid_ask']}/{metrics_h030['n_quotes']} "
          f"martingale_ok={metrics_h030['martingale_constraint_satisfied']} "
          f"elapsed={metrics_h030['elapsed_seconds']:.1f}s")
    print(f"Total self-test wall time: {time.time() - t0:.1f}s\n")

    print("Paper-reported values (Table 4), NOT recomputed here -- for reference only:")
    header = ("key", "sigma0", "theta", "kappa1", "kappa2", "beta", "eps", "rho",
              "kappa2-beta", "gamma", "rms", "inside", "n_quotes")
    print(("{:<12}" + "{:>10}" * (len(header) - 1)).format(*header))
    for key, ref in EXP5_TABLE4_REFERENCE.items():
        print("{:<12}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10d}{:>10d}".format(
            key, ref["sigma0"], ref["theta"], ref["kappa1"], ref["kappa2"], ref["beta"], ref["eps"],
            ref["rho"], ref["kappa2_minus_beta"], ref["gamma"], ref["rms"], ref["inside"], ref["n_quotes"]))

    # Strip non-JSON-serialisable plotting payloads (SmileQuote objects, numpy arrays)
    # before writing the metrics file.
    def _strip(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "_plot_data"}

    out = {
        "synthetic_self_test": [_strip(metrics_h050), _strip(metrics_h030)],
        "table4_paper_reference": EXP5_TABLE4_REFERENCE,
        "note": "table4_paper_reference values are the paper's own published Table 4 numbers, "
                "not recomputed in this replication (no Deribit BTC/ETH quote data available).",
    }
    with open(RESULTS_DIR / "exp5_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    _plot_h050_self_test(metrics_h050, RESULTS_DIR / "exp5_synthetic_smile.png")

    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"\nPeak resident set size: {max_rss_kib} KiB")
    print(f"Wrote {RESULTS_DIR / 'exp5_metrics.json'}")
    print(f"Wrote {RESULTS_DIR / 'exp5_synthetic_smile.png'}")


if __name__ == "__main__":
    main()
