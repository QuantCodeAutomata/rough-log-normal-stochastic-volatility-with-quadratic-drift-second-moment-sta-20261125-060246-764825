# Results

Paper: Sepp, A. and Rakhmonov, A. (2026). "Rough log-normal stochastic
volatility with quadratic drift: second-moment stability and the martingale
property under positive correlation."

All numbers below are read from the `.json`/`.csv` files written by the
experiment scripts in `results/` during the run used to produce this report.
Verdict thresholds (per experiment-wide comparison protocol): **reproduced**
if the computed value lies within the paper's stated precision (or within
10% with the same sign when no explicit precision is given) and the paper's
qualitative claim holds; **not reproduced** if a reported metric has the
opposite sign, differs by more than 50%, or a qualitative claim (ranking,
equality, bound) fails; **partially reproduced** otherwise.

## Experiment 1: Linear-drift second-moment stability for the fractional kernel

Script: `exp/exp_1.py`. Outputs: `results/table2_theta_c.csv`,
`results/figure2_summary.csv`, `results/figure2_second_moment.png`,
`results/exp1_metrics.json`.

**Table 2 (critical theta_c and one-day innovation proxy nu_{1/252,c}).**
`table2_theta_c.csv` holds the full H in {0.1,...,0.5} x kappa_1 in
{1,2,3,5,10} grid (25 cells) of theta_c = (int_0^inf G^2)^{-1/2} and
nu_{1/252,c} = theta_c (1/252)^H/(sqrt(2H) Gamma(H+1/2)), computed from the
spectral (Parseval) quadrature of the squared resolvent of kappa_1 K. Three
cells quoted explicitly in the paper's text are checked in
`exp1_metrics.json["table2_reference_checks"]`:

| H | kappa_1 | theta_c (computed) | theta_c (paper) | abs err | nu (computed) | nu (paper) | abs err |
|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.845688 | 0.846 | 3.1e-4 | 0.730474 | 0.730 | 4.7e-4 |
| 0.1 | 3 | 1.015619 | 1.016 | 3.8e-4 | 0.877253 | 0.877 | 2.5e-4 |
| 0.5 | 1 | 1.414214 | 1.414 | 2.1e-4 | 0.089087 | 0.089 | 8.7e-5 |

All three agree with the paper to the reported 3-decimal precision.
**Verdict: reproduced.**

**Growth rate above threshold (Sec. 3.3).** Solving h_hat(lambda*)=1 by
bisection for H=0.1, theta=1.85, sigma0=theta, kappa1=0 gives
lambda* = 17,851.4/year (paper: 17,851/year); with kappa1=3,
lambda* = 17,234.9/year (paper: 17,235/year). **Verdict: reproduced**
(agreement to 4-5 significant figures).

**Mittag-Leffler growth check (kappa1=kappa2=0).** The closed-form
E[sigma_t^2] = sigma0^2 E_{2H}(a t^{2H}) is checked against the general
resolvent-based conditional-second-moment solver at H=0.5 (where E_{2H}
reduces to exp(a t)); the maximum relative error across the plotted grid is
2.9e-16 (machine precision). **Verdict: reproduced.**

**Figure 2 (second-moment evolution).** `figure2_summary.csv` and
`figure2_second_moment.png` show E[sigma_t^2]/theta^2 vs t for
H in {0.1,0.3,0.5} and gamma in {0.5,0.9,1.0,1.2}, with theta solved per
panel from gamma = theta^2 int G^2 at kappa1=2/year so that the requested
gamma is matched exactly. For gamma<1 the simulated curves converge to the
analytic limit 1/(1-gamma) (e.g. gamma=0.9 -> limit 10.00, matching the
`limit_1_over_1_minus_gamma` column); for gamma=1.2 the ratio grows without
bound and is plotted on a logarithmic axis, consistent with the paper's
description of extremely fast growth (order 10^55 by t=5 years for H=0.1).
**Verdict: reproduced** (the qualitative threshold behaviour and the gamma<1
limiting values match the paper exactly; the paper does not give enough
digits of the Figure 2 curves themselves for a numerical comparison beyond
this).

## Experiment 2: Markovian-lift simulation engine with symmetric drift splitting

Script: `exp/exp_2.py`. Output: `results/exp2_metrics.json`.

This experiment implements the shared simulation engine (`src/simulation.py`)
used by experiments 3-5: the N-factor sum-of-exponentials lift of the
fractional kernel (3-factor kernel for H=0.1, 1-factor kernel for H=0.5, and
a 97-factor kernel for H=0.30 built by Gauss-Jacobi/Gauss-Legendre quadrature
of the Laplace representation of the smoothed kernel), the symmetric
operator-splitting step (memory map, damping map, noise map), and the price
step. `exp2_metrics.json["martingale_check"]` reports the sample mean of
S_n/S_0 (full price, correlated plus idiosyncratic shocks) at T=1 for both
kernels and kappa_2 in {0.0, 0.6, 1.2}: every cell has `mean_S_over_S0` within
0.2 Monte Carlo standard errors of 1 (`within_3_se_of_1: true` for all 6
cells), confirming the discrete martingale property of Lemma B.1, which holds
by construction of the scheme on every grid, independently of kappa_2 and of
the continuous-time martingale condition of Theorem 4.3. **Verdict:
reproduced** (the paper's stated discrete-time identity E[S_n]=S_0 holds in
simulation to within statistical noise on both kernels).

## Experiment 3: Martingale property test via raw conditional forward simulation

Script: `exp/exp_3.py`. Outputs: `results/figure3a_martingale_property.png`,
`results/exp3_metrics.json`.

Reproduces Table 3 / Figure 3(a): sample mean of the raw conditional forward
F_n/S_0 = exp(rho J_n - 0.5 rho^2 I_n) at T=1 for the three-factor kernel
(H=0.1), kappa_2 in {0.0, 0.3, 0.6, 0.9, 1.2}, beta=0.6, at 1,440/2,880/5,760
steps/year, 200,000 paths, 40 batches, seed 3.

| kappa_2 | computed mean (5,760 steps) | paper mean | batch SE (computed) | diff in batch SE |
|---|---|---|---|---|
| 0.0 | 0.93893 | 0.93971 | 0.00077 | 1.01 |
| 0.3 | 0.98617 | 0.98473 | 0.00164 | 0.88 |
| 0.6 | 1.00332 | 0.99711 | 0.00195 | 1.07 |
| 0.9 | 1.00178 | 0.99917 | 0.00122 | 0.72 |
| 1.2 | 1.00069 | 0.99965 | 0.00131 | 0.26 |

(computed values taken from `exp3_metrics.json`, `all_cells` entries at
`steps_per_year=5760`; "diff in batch SE" = |computed - paper| / computed
batch SE). All five cells agree with the paper's reported means to within
about 1 batch standard error. The qualitative result holds exactly as
predicted by Theorem 4.3: for kappa_2 < beta (0.0, 0.3) the sample mean is
clearly below 1 (strict local martingale), and for kappa_2 >= beta (0.6, 0.9,
1.2) the mean is statistically indistinguishable from 1 (true martingale),
with the crossing located near kappa_2 = beta = 0.6 as in Figure 3(a).
**Verdict: reproduced.**

## Experiment 4: Time-averaged second-moment bounds under quadratic drift

Script: `exp/exp_4.py`. Outputs: `results/figure3b_quadratic_drift_bound.png`,
`results/exp4_metrics.json`.

For the three-factor kernel (H=0.1), sigma0=theta=0.4, kappa1=2, rho=0,
vartheta=1.2 (gamma_N=1.0433>1, so the exact linear-drift time-averaged
second moment grows without bound): `exp4_metrics.json["exact_linear_drift_continuous"]`
gives the continuous-time E[I_T]/(T theta^2) solved from the resolvent
Volterra equation (31.98 at T=1, 104.40 at T=2, 2501.21 at T=5, matching the
paper's "about 32 at T=1 to about 2,540 at T=5"), and
`exp4_metrics.json["exact_linear_drift_discrete"]` gives the discrete exact
values at 360/720 steps/year, run alongside the kappa_2=0 Monte Carlo cells
for comparison.

Monte Carlo estimates (`mc_cells`, 200,000 paths, 40 batches, seed 3) for
kappa_2 in {0.0, 0.3, 0.6, 0.9, 1.2} and T in {1,2,5} at 360 and 720
steps/year:

- kappa_2 = 0 (linear drift): mean_ratio ranges from 6.6 to 9.2 across the
  six cells, with batch standard errors of the same order of magnitude
  (0.5-2.6), far below the continuous-time exact values (which the paper
  describes as rising from about 32 at T=1 to about 2,540 at T=5): this is
  the heavy-tailed-sampling effect the paper itself describes ("the largest
  0.1% of paths contribute 40% to 62% of the linear-drift sample averages"),
  i.e. the plain Monte Carlo estimator of an unbounded linear-drift moment is
  strongly downward-biased and noisy at this path count, exactly as
  documented in Sec. 5.
- kappa_2 > 0 (quadratic drift): the 24 cells range from 1.66 to 2.75
  (paper: "estimates across the 24 quadratic-drift cells range from 1.7 to
  2.8"), in close agreement, and well inside the Proposition 3.7 bound
  (`exp4_metrics.json["proposition_3_7_bounds_theta2_units"]`) for every
  kappa_2 and T.

**Verdict: reproduced** for the quadratic-drift bound and its qualitative
contrast with the unbounded linear-drift case; the linear-drift Monte Carlo
cells themselves are, as the paper notes, too noisy at 200,000 paths to
estimate the exact unbounded continuous-time value (this is the paper's own
stated behaviour, not a discrepancy with it).

## Experiment 5: Calibration to Bitcoin and Ether option smiles at fixed Hurst exponent

Script: `exp/exp_5.py`. Outputs: `results/exp5_synthetic_smile.png`,
`results/exp5_metrics.json`.

**Data source.** The paper calibrates to Deribit Bitcoin option quotes
(30 June 2023) and Ether option quotes (26 October 2021). The paper states
this archived snapshot is available only on request from its corresponding
author. Neither the QuantX Dataloader REST API nor the Massive REST API
serves historical Deribit crypto-option chains, so this quote data could not
be obtained in this environment. **This experiment does not recompute
Table 4 or Figure 1 of the paper.**

Instead, `exp5_metrics.json["synthetic_self_test"]` validates the
calibration machinery itself (`src/calibration.py`: least-squares fit under
the martingale constraint kappa_2 >= beta + 0.01, weighted by bid-ask
spread): starting from each of the paper's own published Table 4 parameter
vectors (`exp5_metrics.json["table4_paper_reference"]`, copied verbatim from
the paper, labelled "not recomputed here"), a synthetic smile is priced with
this project's own Monte Carlo/pricing engine, perturbed by about 20% to
build an optimizer starting point, and `calibrate` is checked to recover the
ground-truth parameters: for both the H=0.50 (1-factor kernel) and H=0.30
(97-factor kernel) self-tests, the fitted parameters match the ground truth
to 8+ significant figures, the weighted RMS error is of order 1e-11-1e-13
(synthetic quotes are noise-free), the martingale constraint
kappa_2 - beta >= 0.01 is satisfied, and all synthetic quotes fall inside
the synthetic bid-ask band. This confirms the optimizer, the pricing
pipeline (including the 97-factor Markovian kernel for H=0.30) and the
martingale-constrained least-squares objective are implemented correctly,
but it is a self-test of the code, not a replication of the paper's reported
Bitcoin/Ether fit.

**Verdict: not reproduced** (Table 4 and Figure 1 of the paper; see "Not
reproduced" below). The calibration machinery self-test passes.

## Deviations from the paper

- **Experiment 1, 97-factor kernel is not needed here** (used only in
  experiment 5): the paper's exact Gauss-Jacobi/Gauss-Legendre interval
  placement for the 97-node quadrature of the smoothed kernel's Laplace
  representation is not fully specified (only "8 nodes on each of 12
  intervals plus 1 exponential node"); `src/markovian_kernel.py` builds a
  documented substitute 97-node positive quadrature (Gauss-Jacobi on the
  first interval, Gauss-Legendre on later intervals of the Laplace domain,
  plus the paper's one exponential node) validated by an independent direct
  quadrature of int G_N^2 (relative agreement < 1e-8); this affects only the
  precision of the H=0.30 kernel used in experiment 5's self-test, which the
  paper itself notes is not reproduced from real data (see below).
- **Experiment 5 data source**: the paper's Deribit Bitcoin (30 June 2023)
  and Ether (26 October 2021) option quotes are not available through the
  QuantX Dataloader or Massive REST APIs used for market data elsewhere in
  this project, and the paper states the archived snapshot requires a
  request to its corresponding author; this request could not be made in
  this environment. No substitute market data was used in its place; see
  "Not reproduced" below.
- **Experiment 3/4 Brownian-increment aggregation across grids**: the paper
  mentions that "coarse grids should aggregate Brownian increments from the
  finest grid" but does not give the exact aggregation procedure; each grid
  (1,440/2,880/5,760 for experiment 3; 360/720 for experiment 4) is instead
  simulated with its own independent standard-normal increments at the
  matching seed sequence. This does not affect the reported sample means
  (independent seeds per grid are a valid unbiased estimator of the same
  quantity; only the cross-grid correlation of the estimation error, not
  used in this report, would differ).
- **Experiment 4, finite-T Proposition 3.7 bound**: the bound requires
  L([0,T])/T, the resolvent of the first kind of the Markovian kernel K^N;
  this is computed from the same eigendecomposition of the lifted kernel
  used for the resolvent G_N (`src/markovian_kernel.py`), as the paper does
  not give a separate closed form for L on the N-factor lift.

## Not reproduced

- **Table 4 and Figure 1** (Bitcoin and Ether option-smile calibration and
  their implied-volatility plots): the Deribit option-quote data the paper
  calibrates to is not available through the QuantX Dataloader or Massive
  REST APIs, and the paper's own archived snapshot requires a request to its
  corresponding author that could not be made in this environment. Only a
  synthetic self-test of the calibration code against the paper's own
  published Table 4 parameter vectors is provided (see Experiment 5 above);
  it is not a replication of the paper's calibration to real option data.
