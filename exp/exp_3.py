"""Experiment 3: martingale property test via raw conditional forward
simulation (paper Sec. 4.2, Table 3, Figure 3(a)).

For the three-factor kernel at H=0.1, Theorem 4.3 states that the continuous-
time price process is a true martingale when kappa_2 >= beta and a strict
local martingale when kappa_2 < beta. The discrete scheme itself is an exact
martingale on every grid (Lemma B.1) for the *full* price S_n = exp(X_n),
which combines the correlated and idiosyncratic shocks; the strict-local-
martingale phenomenon instead shows up in the *raw conditional forward*
F_n/S_0 = exp(rho J_n - 0.5 rho^2 I_n), obtained by conditioning on the
common-factor path only (Appendix B.3, Eq. B6). This experiment follows the
paper's own diagnostic and reports the sample mean of F_n/S_0 at T=1 for
kappa_2 in {0.0, 0.3, 0.6, 0.9, 1.2} across grid refinements.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP3_BETA, EXP3_EPS, EXP3_H, EXP3_KAPPA1, EXP3_KAPPA2_GRID, EXP3_N_BATCHES,
    EXP3_N_PATHS_RUN, EXP3_SEED, EXP3_SIGMA0, EXP3_STEPS_PER_YEAR_GRID_RUN,
    EXP3_T, EXP3_THETA, KERNEL_3F_NODES, KERNEL_3F_WEIGHTS,
)
from src.markovian_kernel import KernelSpec
from src.simulation import ModelParams, simulate_paths

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Reference values from Table 3 of the paper, at 5,760 steps/year (Sec. 4.2).
PAPER_TABLE3_5760 = {
    0.0: (0.93971, 0.00068),
    0.3: (0.98473, 0.00155),
    0.6: (0.99711, 0.00177),
    0.9: (0.99917, 0.00137),
    1.2: (0.99965, 0.00111),
}


def raw_conditional_forward_cell(kernel: KernelSpec, kappa2: float, steps_per_year: int,
                                  n_paths: int, n_batches: int, seed: int) -> dict:
    """One (kappa_2, grid) cell of Table 3: sample mean, per-path standard
    error and batch standard error of the raw conditional forward F_n/S_0
    = exp(rho J_n - 0.5 rho^2 I_n) at T=1 (Appendix B.3, Eq. B6)."""
    params = ModelParams(sigma0=EXP3_SIGMA0, kappa1=EXP3_KAPPA1, kappa2=kappa2,
                          theta=EXP3_THETA, beta=EXP3_BETA, eps=EXP3_EPS)
    out = simulate_paths(kernel, params, T=EXP3_T, steps_per_year=steps_per_year,
                          n_paths=n_paths, seed=seed)
    rho = out["rho"]
    F = np.exp(rho * out["J"] - 0.5 * rho ** 2 * out["I"])

    mean_F = float(np.mean(F))
    per_path_se = float(np.std(F, ddof=1) / np.sqrt(n_paths))
    batch_size = n_paths // n_batches
    batch_means = F[: batch_size * n_batches].reshape(n_batches, batch_size).mean(axis=1)
    batch_se = float(np.std(batch_means, ddof=1) / np.sqrt(n_batches))
    return {"kappa2": float(kappa2), "steps_per_year": steps_per_year, "n_paths": n_paths,
            "mean_F": mean_F, "per_path_se": per_path_se, "batch_se": batch_se}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    kernel = KernelSpec(nodes=KERNEL_3F_NODES, weights=KERNEL_3F_WEIGHTS)

    rows = []
    t_start = time.perf_counter()
    for steps_per_year in EXP3_STEPS_PER_YEAR_GRID_RUN:
        for kappa2 in EXP3_KAPPA2_GRID:
            t0 = time.perf_counter()
            cell = raw_conditional_forward_cell(kernel, kappa2, steps_per_year,
                                                 EXP3_N_PATHS_RUN, EXP3_N_BATCHES, EXP3_SEED)
            cell["seconds"] = time.perf_counter() - t0
            rows.append(cell)
            print(f"steps/yr={steps_per_year:5d} kappa2={kappa2:.1f}: "
                  f"mean F={cell['mean_F']:.5f} +/- {cell['batch_se']:.5f} (batch SE) "
                  f"[{cell['seconds']:.1f}s]")
    total_seconds = time.perf_counter() - t_start

    fine = [r for r in rows if r["steps_per_year"] == EXP3_STEPS_PER_YEAR_GRID_RUN[-1]]
    comparison = []
    for r in fine:
        paper_mean, paper_se = PAPER_TABLE3_5760[r["kappa2"]]
        diff_in_se = abs(r["mean_F"] - paper_mean) / paper_se
        comparison.append({
            "kappa2": r["kappa2"], "run_mean_F": r["mean_F"], "run_batch_se": r["batch_se"],
            "paper_mean_F": paper_mean, "paper_batch_se": paper_se,
            "diff_in_paper_se": diff_in_se,
        })
        print(f"kappa2={r['kappa2']:.1f}: run={r['mean_F']:.5f}, paper={paper_mean:.5f}, "
              f"diff={diff_in_se:.2f} paper-SE")

    martingale_crosses_near_beta = (fine[0]["mean_F"] < 1.0 and
                                     fine[-1]["mean_F"] > fine[0]["mean_F"])
    assert martingale_crosses_near_beta, (
        "Raw conditional forward mean should increase towards 1 as kappa_2 crosses beta"
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    kappa2_vals = [r["kappa2"] for r in fine]
    means = [r["mean_F"] for r in fine]
    errs = [r["batch_se"] for r in fine]
    ax.errorbar(kappa2_vals, means, yerr=errs, fmt="o-", capsize=4, label="simulated mean")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label=r"$S_0=1$")
    ax.axvline(EXP3_BETA, color="red", linestyle=":", linewidth=1, label=r"$\kappa_2=\beta$")
    ax.set_xlabel(r"$\kappa_2$")
    ax.set_ylabel(r"$E[F_T/S_0]$ (raw conditional forward)")
    ax.set_title(f"Figure 3(a): martingale property at {EXP3_STEPS_PER_YEAR_GRID_RUN[-1]} steps/year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3a_martingale_property.png", dpi=150)
    plt.close(fig)

    metrics = {
        "all_cells": rows,
        "comparison_at_finest_grid": comparison,
        "total_seconds": total_seconds,
        "assumption": "Each (kappa2, grid) cell uses an independent random stream; "
                       "the paper's own aggregation of shared Brownian increments across "
                       "grids is not implemented (see results/RESULTS.md).",
    }
    (RESULTS_DIR / "exp3_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"exp_3 complete in {total_seconds:.1f}s; metrics saved to results/exp3_metrics.json")


if __name__ == "__main__":
    main()
