"""Experiment 2: Markovian-lifted simulation engine and the discrete
martingale property (paper Appendix B, Lemma B.1).

This experiment does not reproduce a numbered paper table or figure; it
validates the shared simulation engine (``src/simulation.py``,
``src/markovian_kernel.py``) used by exp_3, exp_4 and exp_5:

1. builds the three-factor kernel (H=0.1) and the single-factor kernel
   (H=0.5) from the paper's published nodes/weights, and the 97-factor
   kernel (H=0.30) used for the Bitcoin/Ether calibration (Appendix B.4);
2. simulates the symmetric-splitting scheme (Eq. B1-B5) across a grid of
   quadratic-damping values kappa_2 and checks the discrete martingale
   property E[S_n] = S_0 (Lemma B.1), which holds by construction of the
   scheme on every time grid, independent of the strict-local-martingale
   question examined in exp_3;
3. times one simulation batch to calibrate the runtime estimates used to
   size exp_3, exp_4 and exp_5.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.config import (
    EXP2_DEMO_BETA, EXP2_DEMO_EPS, EXP2_DEMO_KAPPA1, EXP2_DEMO_KAPPA2_VALUES,
    EXP2_DEMO_N_PATHS, EXP2_DEMO_N_STEPS_PER_YEAR, EXP2_DEMO_SIGMA0, EXP2_DEMO_T,
    EXP2_DEMO_THETA, EXP2_SEED, KERNEL_1F_NODES, KERNEL_1F_WEIGHTS,
    KERNEL_3F_NODES, KERNEL_3F_WEIGHTS, KERNEL_97F_H,
)
from src.markovian_kernel import KernelSpec, build_97_factor_kernel
from src.simulation import ModelParams, simulate_paths

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def martingale_check(kernel: KernelSpec, kernel_name: str, n_paths: int,
                      steps_per_year: int, T: float, seed: int) -> list[dict]:
    """E[S_n]/S_0 - 1 and its Monte Carlo standard error for every kappa_2 in
    ``EXP2_DEMO_KAPPA2_VALUES`` (Lemma B.1: the discrete scheme is an exact
    martingale on every grid by construction of the price step Eq. B5)."""
    rows = []
    for kappa2 in EXP2_DEMO_KAPPA2_VALUES:
        params = ModelParams(sigma0=EXP2_DEMO_SIGMA0, kappa1=EXP2_DEMO_KAPPA1,
                              kappa2=kappa2, theta=EXP2_DEMO_THETA,
                              beta=EXP2_DEMO_BETA, eps=EXP2_DEMO_EPS)
        t0 = time.perf_counter()
        out = simulate_paths(kernel, params, T=T, steps_per_year=steps_per_year,
                              n_paths=n_paths, seed=seed)
        elapsed = time.perf_counter() - t0
        S = np.exp(out["X"])
        mean_S = float(np.mean(S))
        se_S = float(np.std(S, ddof=1) / np.sqrt(n_paths))
        rows.append({
            "kernel": kernel_name, "kappa2": float(kappa2), "n_paths": n_paths,
            "steps_per_year": steps_per_year, "T": T, "mean_S_over_S0": mean_S,
            "mc_std_error": se_S, "within_3_se_of_1": bool(abs(mean_S - 1.0) <= 3 * se_S),
            "seconds": elapsed,
        })
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    kernel_3f = KernelSpec(nodes=KERNEL_3F_NODES, weights=KERNEL_3F_WEIGHTS)
    kernel_1f = KernelSpec(nodes=KERNEL_1F_NODES, weights=KERNEL_1F_WEIGHTS)

    rows = []
    rows += martingale_check(kernel_3f, "3-factor_H0.1", EXP2_DEMO_N_PATHS,
                              EXP2_DEMO_N_STEPS_PER_YEAR, EXP2_DEMO_T, EXP2_SEED)
    rows += martingale_check(kernel_1f, "1-factor_H0.5", EXP2_DEMO_N_PATHS,
                              EXP2_DEMO_N_STEPS_PER_YEAR, EXP2_DEMO_T, EXP2_SEED)

    for r in rows:
        print(f"{r['kernel']} kappa2={r['kappa2']}: E[S_T]/S_0={r['mean_S_over_S0']:.5f} "
              f"+/- {r['mc_std_error']:.5f} (within 3 SE of 1: {r['within_3_se_of_1']})")
    assert all(r["within_3_se_of_1"] for r in rows), (
        "Discrete martingale property (Lemma B.1) violated beyond Monte Carlo noise"
    )

    kernel_97f = build_97_factor_kernel(H=KERNEL_97F_H)
    assert kernel_97f.n_factors == 97, "97-factor kernel must have exactly 97 nodes (Eq. B9)"
    assert np.all(kernel_97f.weights > 0), "97-factor kernel weights must be positive (Eq. B9)"
    kernel_summary = {
        "n_factors_3f": kernel_3f.n_factors, "wbar_3f": kernel_3f.wbar,
        "n_factors_1f": kernel_1f.n_factors, "wbar_1f": kernel_1f.wbar,
        "n_factors_97f": kernel_97f.n_factors, "wbar_97f": float(kernel_97f.wbar),
    }
    print(f"Kernel summary: {kernel_summary}")

    one_path_seconds = rows[-1]["seconds"] / rows[-1]["n_paths"] / rows[-1]["steps_per_year"]
    metrics = {
        "martingale_check": rows,
        "kernel_summary": kernel_summary,
        "seconds_per_path_per_step": one_path_seconds,
    }
    (RESULTS_DIR / "exp2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"exp_2 complete; seconds/path/step = {one_path_seconds:.3e} "
          f"(used to size exp_3/exp_4/exp_5 runtimes)")
    print("Metrics saved to results/exp2_metrics.json")


if __name__ == "__main__":
    main()
