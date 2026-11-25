"""Experiment 4: time-averaged second-moment bounds under quadratic drift
(paper Sec. 3.4, Sec. 5, Figure 3(b)).

For the three-factor Markovian-lifted kernel at H=0.1 the linear-drift
(kappa_2=0) stability parameter gamma_N = theta_vol^2 * int G_N^2 exceeds 1,
so the exact linear-drift time-averaged second moment E[I_T]/(T theta_vol^2)
grows without bound as T increases (Sec. 3.3-3.4). Proposition 3.7 shows that
adding quadratic damping (kappa_2>0) bounds E[I_T]/T uniformly, independently
of theta_vol and H. This experiment computes, for kappa_2 in
{0.0, 0.3, 0.6, 0.9, 1.2} and T in {1, 2, 5} years:
  - the exact linear-drift continuous-time and discrete-grid second moment
    (kappa_2=0 only), via the Volterra product-integration solver of
    src.bounds.exact_linear_drift_second_moment;
  - the Proposition 3.7 asymptotic bound on E[I_T]/T (kappa_2>0 only), via
    src.bounds.proposition_3_7_bound;
  - Monte Carlo estimates of E[I_T]/(T theta_vol^2) for every kappa_2, at
    360 and 720 steps/year, using the shared symmetric-splitting engine
    (src.simulation.simulate_paths).
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
    EXP4_KAPPA1, EXP4_KAPPA2_GRID, EXP4_N_BATCHES, EXP4_N_PATHS_RUN,
    EXP4_RHO, EXP4_SEED, EXP4_SIGMA0, EXP4_STEPS_PER_YEAR_GRID_RUN,
    EXP4_T_GRID, EXP4_THETA, EXP4_VARTHETA, KERNEL_3F_NODES, KERNEL_3F_WEIGHTS,
)
from src.markovian_kernel import KernelSpec, resolvent_G_N_modes, integral_G_N_squared, ell_N
from src.bounds import exact_linear_drift_second_moment, proposition_3_7_bound
from src.simulation import ModelParams, simulate_paths

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def mc_cell(kernel: KernelSpec, kappa2: float, T: float, steps_per_year: int,
            n_paths: int, n_batches: int, seed: int) -> dict:
    """E[I_T]/(T theta_vol^2) Monte Carlo cell (Sec. 3.4, Figure 3(b))."""
    # EXP4_RHO=0 (beta=0): ModelParams.theta_vol = sqrt(beta^2+eps^2) = eps = EXP4_VARTHETA.
    params = ModelParams(sigma0=EXP4_SIGMA0, kappa1=EXP4_KAPPA1, kappa2=kappa2,
                          theta=EXP4_THETA, beta=EXP4_RHO * EXP4_VARTHETA, eps=EXP4_VARTHETA)
    out = simulate_paths(kernel, params, T=T, steps_per_year=steps_per_year,
                          n_paths=n_paths, seed=seed)
    ratio = out["I"] / (T * params.theta_vol ** 2)
    mean_r = float(np.mean(ratio))
    batch_size = n_paths // n_batches
    batch_means = ratio[: batch_size * n_batches].reshape(n_batches, batch_size).mean(axis=1)
    batch_se = float(np.std(batch_means, ddof=1) / np.sqrt(n_batches))
    return {"kappa2": float(kappa2), "T": float(T), "steps_per_year": steps_per_year,
            "mean_ratio": mean_r, "batch_se": batch_se}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    kernel = KernelSpec(nodes=KERNEL_3F_NODES, weights=KERNEL_3F_WEIGHTS)

    poles, residues = resolvent_G_N_modes(kernel, EXP4_KAPPA1)
    int_G2 = integral_G_N_squared(poles, residues)
    gamma_N = float(EXP4_VARTHETA ** 2 * int_G2)
    ell = float(ell_N(kernel))
    print(f"gamma_N (3-factor, H={EXP4_T_GRID is not None and 0.1}) = {gamma_N:.4f}  ell_N = {ell:.5f}")
    assert gamma_N > 1.0, "The three-factor kernel at H=0.1 must have gamma_N > 1 (Sec. 3.4)"

    # Exact linear-drift (kappa2=0) continuous and discrete time-averaged second moment.
    exact_continuous = exact_linear_drift_second_moment(
        EXP4_T_GRID, EXP4_SIGMA0, EXP4_THETA, EXP4_KAPPA1, kernel, EXP4_VARTHETA, n_grid=8000
    ) / (EXP4_T_GRID * EXP4_THETA ** 2)
    exact_discrete = {
        ng: (exact_linear_drift_second_moment(
                EXP4_T_GRID, EXP4_SIGMA0, EXP4_THETA, EXP4_KAPPA1, kernel, EXP4_VARTHETA, n_grid=ng
             ) / (EXP4_T_GRID * EXP4_THETA ** 2)).tolist()
        for ng in EXP4_STEPS_PER_YEAR_GRID_RUN
    }
    print("Exact linear-drift E[I_T]/(T theta^2), continuous:", exact_continuous)
    for ng, vals in exact_discrete.items():
        print(f"  discrete grid {ng} steps/yr:", vals)

    # Proposition 3.7 bounds, expressed in E[I_T]/(T theta_vol^2) units for comparability
    # with the simulated ratio (bound is independent of theta_vol; dividing by
    # EXP4_THETA**2 here follows the paper's own theta^2-normalization of Figure 3(b)).
    prop37_bounds = {}
    for kappa2 in EXP4_KAPPA2_GRID:
        if kappa2 <= 0:
            continue
        raw_bound = proposition_3_7_bound(EXP4_KAPPA1, float(kappa2), EXP4_THETA, EXP4_SIGMA0, kernel)
        prop37_bounds[float(kappa2)] = raw_bound / EXP4_THETA ** 2
    print("Proposition 3.7 bounds (E[I_T]/(T theta^2) units):", prop37_bounds)

    # Monte Carlo cells.
    rows = []
    t_start = time.perf_counter()
    for steps_per_year in EXP4_STEPS_PER_YEAR_GRID_RUN:
        for kappa2 in EXP4_KAPPA2_GRID:
            for T in EXP4_T_GRID:
                t0 = time.perf_counter()
                cell = mc_cell(kernel, float(kappa2), float(T), steps_per_year,
                                EXP4_N_PATHS_RUN, EXP4_N_BATCHES, EXP4_SEED)
                cell["seconds"] = time.perf_counter() - t0
                rows.append(cell)
                print(f"steps/yr={steps_per_year:4d} kappa2={kappa2:.1f} T={T:.0f}: "
                      f"E[I_T]/(T th^2)={cell['mean_ratio']:.3f} +/- {cell['batch_se']:.3f} "
                      f"[{cell['seconds']:.1f}s]")
    total_seconds = time.perf_counter() - t_start

    quadratic_cells = [r for r in rows if r["kappa2"] > 0]
    quad_ratios = [r["mean_ratio"] for r in quadratic_cells]
    print(f"Quadratic-drift (kappa2>0) cell range: "
          f"[{min(quad_ratios):.2f}, {max(quad_ratios):.2f}] "
          f"(paper range: 1.7 to 2.8, Sec. 5)")

    # Sanity check: every quadratic-drift MC cell must lie (well) below its
    # Proposition 3.7 bound (Sec. 3.4, Eq. (20)/(22)).
    for r in quadratic_cells:
        bound = prop37_bounds[r["kappa2"]]
        assert r["mean_ratio"] < bound, (
            f"MC estimate {r['mean_ratio']:.3f} at kappa2={r['kappa2']} exceeds the "
            f"Proposition 3.7 bound {bound:.3f}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {360: "o", 720: "s"}
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(EXP4_KAPPA2_GRID)))
    for color, kappa2 in zip(colors, EXP4_KAPPA2_GRID):
        if kappa2 == 0:
            continue
        for ng in EXP4_STEPS_PER_YEAR_GRID_RUN:
            cells = [r for r in rows if r["kappa2"] == kappa2 and r["steps_per_year"] == ng]
            cells.sort(key=lambda r: r["T"])
            ax.plot([c["T"] for c in cells], [c["mean_ratio"] for c in cells],
                     marker=markers[ng], linestyle="-", color=color,
                     label=f"MC $\\kappa_2$={kappa2:.1f}, {ng}/yr" if ng == 720 else None)
        ax.axhline(prop37_bounds[kappa2], color=color, linestyle="--", linewidth=1)
    zero_cells = [r for r in rows if r["kappa2"] == 0 and r["steps_per_year"] == EXP4_STEPS_PER_YEAR_GRID_RUN[-1]]
    zero_cells.sort(key=lambda r: r["T"])
    ax.plot([c["T"] for c in zero_cells], [c["mean_ratio"] for c in zero_cells],
             "k:", marker="x", label=r"MC $\kappa_2$=0")
    ax.plot(EXP4_T_GRID, exact_continuous, "kD", label="exact linear-drift (continuous)")
    ax.set_yscale("log")
    ax.set_xlabel("T (years)")
    ax.set_ylabel(r"$E[I_T]/(T\theta^2)$")
    ax.set_title("Figure 3(b): time-averaged second moment under quadratic drift")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figure3b_quadratic_drift_bound.png", dpi=150)
    plt.close(fig)

    metrics = {
        "gamma_N_3factor_H0.1": gamma_N,
        "ell_N_3factor_H0.1": ell,
        "exact_linear_drift_continuous": {str(T): float(v) for T, v in zip(EXP4_T_GRID, exact_continuous)},
        "exact_linear_drift_discrete": {str(ng): vals for ng, vals in exact_discrete.items()},
        "proposition_3_7_bounds_theta2_units": prop37_bounds,
        "mc_cells": rows,
        "quadratic_cell_range": [min(quad_ratios), max(quad_ratios)],
        "paper_quadratic_cell_range": [1.7, 2.8],
        "total_seconds": total_seconds,
    }
    (RESULTS_DIR / "exp4_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"exp_4 complete in {total_seconds:.1f}s; metrics saved to results/exp4_metrics.json")


if __name__ == "__main__":
    main()
