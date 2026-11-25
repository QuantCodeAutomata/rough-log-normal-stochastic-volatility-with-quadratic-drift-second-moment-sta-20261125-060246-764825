"""Experiment 1: linear-drift (kappa2=0) second-moment stability analysis for
the Riemann-Liouville fractional kernel (paper Sec. 3.1-3.3).

Reproduces:
    - Table 2: critical volatility-of-volatility theta_c and the one-day
      innovation proxy nu_{1/252,c}, for H in {0.1,...,0.5} and
      kappa_1 in {1,2,3,5,10} (Sec. 3.3, Table 2 on page 29).
    - Figure 2: the conditional second moment E[sigma_t^2]/theta^2 as a
      function of time, for H in {0.1,0.3,0.5} and the stability parameter
      gamma in {0.5,0.9,1.0,1.2} (Sec. 3.3, Figure 2).
    - The exponential growth rate lambda* above the stability threshold
      (Sec. 3.3 text) and the Mittag-Leffler growth formula for kappa1=0
      (Corollary 3.3).

Run: python -m exp.exp_1
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    EXP1_DT_INNOVATION,
    EXP1_FIG2_GAMMA_VALUES,
    EXP1_FIG2_H_VALUES,
    EXP1_FIG2_KAPPA1,
    EXP1_FIG2_N_STEPS_PER_YEAR,
    EXP1_FIG2_SIGMA0_OVER_THETA,
    EXP1_FIG2_T_MAX,
    EXP1_H_GRID,
    EXP1_KAPPA1_GRID,
    EXP1_LAMBDA_STAR_H,
    EXP1_LAMBDA_STAR_KAPPA1_VALUES,
    EXP1_LAMBDA_STAR_THETA,
)
from src.kernels import (
    ResolventTable,
    alpha_of_H,
    conditional_second_moment,
    gamma_param,
    growth_rate_lambda_star,
    innovation_proxy,
    integral_G_squared,
    mittag_leffler_growth_curve,
    theta_c,
)

RESULTS_DIR = "results"

# Table 2 reference values quoted in the paper (Sec. 3.3, Table 2), used only
# to log the reproduction error, never to overwrite computed numbers.
TABLE2_PAPER_REFERENCE = {
    (0.1, 1.0): (0.846, 0.730),
    (0.1, 3.0): (1.016, 0.877),
    (0.5, 1.0): (1.414, 0.089),
}


def build_table2() -> pd.DataFrame:
    """Table 2: theta_c and nu_{1/252,c} over the (H, kappa1) grid (Sec. 3.3)."""
    rows = []
    for H in EXP1_H_GRID:
        alpha = alpha_of_H(H)
        for kappa1 in EXP1_KAPPA1_GRID:
            tc = theta_c(alpha, kappa1)
            nu = innovation_proxy(tc, H, EXP1_DT_INNOVATION)
            rows.append({"H": H, "kappa1": kappa1, "theta_c": tc, "nu_1_252_c": nu})
    return pd.DataFrame(rows)


def log_table2_reference_check(table2: pd.DataFrame) -> list[dict]:
    """Compares a subset of Table 2 against the paper's reported values."""
    checks = []
    for (H, kappa1), (tc_ref, nu_ref) in TABLE2_PAPER_REFERENCE.items():
        row = table2[(np.isclose(table2["H"], H)) & (np.isclose(table2["kappa1"], kappa1))].iloc[0]
        checks.append({
            "H": H, "kappa1": kappa1,
            "theta_c_computed": row["theta_c"], "theta_c_paper": tc_ref,
            "nu_computed": row["nu_1_252_c"], "nu_paper": nu_ref,
            "theta_c_abs_err": abs(row["theta_c"] - tc_ref),
            "nu_abs_err": abs(row["nu_1_252_c"] - nu_ref),
        })
        print(f"H={H}, kappa1={kappa1}: theta_c={row['theta_c']:.4f} (paper {tc_ref}), "
              f"nu={row['nu_1_252_c']:.4f} (paper {nu_ref})")
    return checks


def build_figure2() -> None:
    """Figure 2: E[sigma_t^2]/theta^2 vs t for H in {0.1,0.3,0.5}, gamma in
    {0.5,0.9,1.0,1.2} (Sec. 3.3). kappa_1 is held fixed; theta is solved from
    the target gamma = theta^2 int G^2 at that kappa_1 (sigma_0=theta, per the
    Figure 2 caption)."""
    fig, axes = plt.subplots(1, len(EXP1_FIG2_H_VALUES), figsize=(15, 4.5), sharex=True)
    kappa1 = EXP1_FIG2_KAPPA1
    summary_rows = []

    for ax, H in zip(axes, EXP1_FIG2_H_VALUES):
        alpha = alpha_of_H(H)
        Gsq = integral_G_squared(alpha, kappa1)
        table = ResolventTable.build(alpha, kappa1, t_max=EXP1_FIG2_T_MAX, n_points=250)
        for gamma in EXP1_FIG2_GAMMA_VALUES:
            theta = float(np.sqrt(gamma / Gsq))
            sigma0 = EXP1_FIG2_SIGMA0_OVER_THETA * theta
            t_grid, _, v = conditional_second_moment(
                table, sigma0=sigma0, theta=theta, T=EXP1_FIG2_T_MAX,
                steps_per_year=EXP1_FIG2_N_STEPS_PER_YEAR,
            )
            ratio = v / theta ** 2
            ax.plot(t_grid, ratio, label=f"gamma={gamma}")
            summary_rows.append({
                "H": H, "gamma": gamma, "theta": theta,
                "ratio_at_t0": ratio[0], "ratio_at_Tmax": ratio[-1],
                "limit_1_over_1_minus_gamma": 1.0 / (1.0 - gamma) if gamma < 1.0 else np.nan,
            })
        ax.set_yscale("log")
        ax.set_xlabel("t (years)")
        ax.set_title(f"H = {H}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"E[$\sigma_t^2$] / $\theta^2$  (log scale)")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 2: conditional second moment under linear drift (kappa2=0, Sec. 3.3)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/figure2_second_moment.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(f"{RESULTS_DIR}/figure2_summary.csv", index=False)
    print(f"Figure 2 saved to {RESULTS_DIR}/figure2_second_moment.png")


def growth_rate_illustration() -> dict:
    """lambda* at H=0.1, theta=1.85 for kappa1 in {0,3} (Sec. 3.3 text), plus
    a check of the Mittag-Leffler growth formula at kappa1=kappa2=0
    (Corollary 3.3)."""
    alpha = alpha_of_H(EXP1_LAMBDA_STAR_H)
    theta = EXP1_LAMBDA_STAR_THETA
    lambdas = {}
    for kappa1 in EXP1_LAMBDA_STAR_KAPPA1_VALUES:
        gamma = gamma_param(theta, alpha, kappa1) if kappa1 > 0 else np.inf
        lam = growth_rate_lambda_star(theta, alpha, kappa1)
        lambdas[kappa1] = lam
        print(f"H={EXP1_LAMBDA_STAR_H}, theta={theta}, kappa1={kappa1}: "
              f"gamma={gamma if np.isfinite(gamma) else 'inf'}, lambda*={lam:.1f} /year")

    # Mittag-Leffler growth check (kappa1=kappa2=0): E[sigma_t^2] = sigma0^2 E_{2H}(a t^{2H}).
    t = np.linspace(0.0, 1.0, 50)
    sigma0 = 0.3
    ml_theta = 0.3
    growth = mittag_leffler_growth_curve(t, H=0.5, sigma0=sigma0, theta=ml_theta)
    expected_classical = sigma0 ** 2 * np.exp(ml_theta ** 2 * t)
    ml_rel_err = float(np.max(np.abs(growth - expected_classical) / expected_classical))
    print(f"Mittag-Leffler growth check at H=0.5 vs exact exponential: max rel err {ml_rel_err:.2e}")

    return {
        "lambda_star_H0.1_kappa1_0": lambdas[EXP1_LAMBDA_STAR_KAPPA1_VALUES[0]],
        "lambda_star_H0.1_kappa1_3": lambdas[EXP1_LAMBDA_STAR_KAPPA1_VALUES[1]],
        "mittag_leffler_growth_max_rel_err_at_H0.5": ml_rel_err,
    }


def main() -> None:
    t0 = time.time()
    table2 = build_table2()
    table2.to_csv(f"{RESULTS_DIR}/table2_theta_c.csv", index=False)
    print(f"Table 2 saved to {RESULTS_DIR}/table2_theta_c.csv ({len(table2)} rows)")
    reference_checks = log_table2_reference_check(table2)

    build_figure2()
    growth = growth_rate_illustration()

    metrics = {
        "table2_reference_checks": reference_checks,
        "growth_rate": growth,
        "runtime_seconds": time.time() - t0,
    }
    with open(f"{RESULTS_DIR}/exp1_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"exp_1 complete in {metrics['runtime_seconds']:.1f}s; metrics saved to "
          f"{RESULTS_DIR}/exp1_metrics.json")


if __name__ == "__main__":
    main()
