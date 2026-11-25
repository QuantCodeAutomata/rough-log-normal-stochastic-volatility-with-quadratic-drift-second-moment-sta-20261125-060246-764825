"""Module-level configuration for every experiment.

All numerical parameters are taken from the paper "Rough log-normal stochastic
volatility with quadratic drift: second-moment stability and the martingale
property under positive correlation" (Sepp and Rakhmonov, 2026), as summarised
in the accompanying experiment specification. Each variable states its
meaning, unit and source section/equation, or "not specified in the paper"
when the paper leaves it open.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Experiment 1: linear-drift second-moment stability for the fractional kernel
# ---------------------------------------------------------------------------

EXP1_H_GRID = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Hurst exponents, Table 2 (Sec. 3.3)
EXP1_KAPPA1_GRID = np.array([1.0, 2.0, 3.0, 5.0, 10.0])  # kappa_1 (1/year), Table 2
EXP1_DT_INNOVATION = 1.0 / 252.0  # Delta for the one-day innovation proxy, Sec. 3.3

EXP1_FIG2_H_VALUES = (0.1, 0.3, 0.5)  # Hurst values plotted in Figure 2
EXP1_FIG2_GAMMA_VALUES = (0.5, 0.9, 1.0, 1.2)  # gamma values plotted in Figure 2
EXP1_FIG2_KAPPA1 = 2.0  # kappa_1 (1/year) used for Figure 2 (Fig. 2 caption, sigma0=theta=... kappa1=2)
EXP1_FIG2_T_MAX = 5.0  # time horizon (years) for Figure 2
EXP1_FIG2_SIGMA0_OVER_THETA = 1.0  # sigma_0 = theta, per the Figure 2 caption
# theta is not pinned numerically beyond sigma_0=theta; for each (H, gamma) panel theta is
# instead solved from gamma = theta^2 int_0^inf G(u)^2 du at the fixed kappa_1 above, which
# reproduces the requested gamma exactly and leaves the plotted ratio E[sigma_t^2]/theta^2
# scale-free in theta (Sec. 3.3).
EXP1_FIG2_N_STEPS_PER_YEAR = 4000  # product-integration steps per year (Fig. 2 caption: 4,000/8,000)
EXP1_FIG2_N_STEPS_PER_YEAR_FINE = 8000  # finer grid used to check convergence (Fig. 2 caption)

EXP1_LAMBDA_STAR_H = 0.1  # Hurst value used for the growth-rate illustration (Sec. 3.3 text)
EXP1_LAMBDA_STAR_THETA = 1.85  # vartheta (year^-H) for the growth-rate illustration (Sec. 3.3 text)
EXP1_LAMBDA_STAR_KAPPA1_VALUES = (0.0, 3.0)  # kappa_1 values compared in the growth-rate illustration

# ---------------------------------------------------------------------------
# Experiment 2: Markovian lift and symmetric-splitting simulation scheme
# ---------------------------------------------------------------------------

# Three-factor kernel at H=0.1, T=1 year, Bayer-Breneis (2023) quadrature rule (Sec. 2.3)
KERNEL_3F_NODES = np.array([0.0333, 2.242, 46.83])  # x_i (1/year)
KERNEL_3F_WEIGHTS = np.array([0.555, 1.111, 6.086])  # w_i (year^{H-1/2})

# Single-factor kernel representing H=1/2 (Sec. 2.3): K=1, N=1, x_1=0, w_1=1
KERNEL_1F_NODES = np.array([0.0])
KERNEL_1F_WEIGHTS = np.array([1.0])

# 97-factor kernel approximating the smoothed fractional kernel at H=0.30 (Appendix B.4, Eq. B9)
KERNEL_97F_H = 0.30  # Hurst exponent of the calibration kernel
KERNEL_97F_T0 = 1.0 / (365.0 * 512.0)  # years; shift removing the kernel singularity at zero (Eq. B9)
KERNEL_97F_B = 0.0712  # exponential-tail coefficient b in Eq. (B9)
KERNEL_97F_N_INTERVALS = 12  # number of Laplace-domain intervals (Appendix B.4: "12 intervals")
KERNEL_97F_N_NODES_PER_INTERVAL = 8  # Gaussian-quadrature nodes per interval (Appendix B.4: "8 nodes")
# Appendix B.4 states the 97-factor kernel uses 8 nodes on each of 12 intervals plus one
# exponential node (8*12+1=97); the exact placement of the 12 intervals in the Laplace
# domain is not specified in the paper and is implemented here as a log-spaced quadrature
# of the completely monotone kernel (B9), documented as an assumption in results/RESULTS.md.

EXP2_DEMO_SIGMA0 = 0.4  # initial volatility (decimal units), illustrative value for the engine demo
EXP2_DEMO_THETA = 0.4  # reference volatility level theta (decimal units)
EXP2_DEMO_KAPPA1 = 2.0  # linear mean-reversion speed kappa_1 (1/year)
EXP2_DEMO_KAPPA2_VALUES = (0.0, 0.6, 1.2)  # quadratic damping kappa_2 values used in the engine check
EXP2_DEMO_BETA = 0.3  # volatility beta (spot-volatility covariance loading)
EXP2_DEMO_EPS = 0.6  # idiosyncratic volatility-of-volatility loading epsilon
EXP2_DEMO_N_PATHS = 20000  # Monte Carlo paths for the martingale sanity check
EXP2_DEMO_N_STEPS_PER_YEAR = 365  # steps per year for the engine demonstration
EXP2_DEMO_T = 1.0  # horizon in years for the engine demonstration
EXP2_SEED = 3  # random seed (Table 3 caption: "seed 3")

# ---------------------------------------------------------------------------
# Experiment 3: martingale property via raw conditional forward simulation
# ---------------------------------------------------------------------------

EXP3_H = 0.1  # Hurst exponent, Table 3
EXP3_BETA = 0.6  # volatility beta, Table 3 / Figure 3 caption
EXP3_EPS = 0.8  # idiosyncratic vol-of-vol loading, Table 3 / Figure 3 caption
EXP3_SIGMA0 = 0.5  # initial volatility, Table 3
EXP3_THETA = 0.4  # reference volatility level, Table 3
EXP3_KAPPA1 = 2.0  # linear mean-reversion speed (1/year), Table 3
EXP3_KAPPA2_GRID = np.array([0.0, 0.3, 0.6, 0.9, 1.2])  # quadratic damping values, Table 3
EXP3_STEPS_PER_YEAR_GRID = (1440, 2880, 5760)  # time grids, Table 3
EXP3_T = 1.0  # horizon in years, Table 3
EXP3_N_PATHS_PAPER = 200_000  # paths per cell in the paper, Table 3 caption
EXP3_N_BATCHES = 40  # number of batches of 5,000 paths each, Table 3 caption
EXP3_SEED = 3  # random seed, Table 3 caption

# Reduced run executed in this replication (see results/RESULTS.md for the runtime estimate
# that led to this reduction and the --full flag that restores the paper configuration).
EXP3_N_PATHS_RUN = 200_000  # paths per cell actually executed
EXP3_STEPS_PER_YEAR_GRID_RUN = (1440, 2880, 5760)  # grids actually executed

# ---------------------------------------------------------------------------
# Experiment 4: time-averaged second moment bounds under quadratic drift
# ---------------------------------------------------------------------------

EXP4_H = 0.1  # Hurst exponent (three-factor kernel), Figure 3(b) caption
EXP4_SIGMA0 = 0.4  # initial volatility = theta, Figure 3(b) caption
EXP4_THETA = 0.4  # reference volatility level, Figure 3(b) caption
EXP4_KAPPA1 = 2.0  # linear mean-reversion speed (1/year), Figure 3(b) caption
EXP4_RHO = 0.0  # spot-volatility correlation (beta=0), Figure 3(b) caption
EXP4_VARTHETA = 1.2  # total volatility-of-volatility (year^-H), Figure 3(b) caption
EXP4_KAPPA2_GRID = np.array([0.0, 0.3, 0.6, 0.9, 1.2])  # quadratic damping values, Figure 3(b)
EXP4_T_GRID = np.array([1.0, 2.0, 5.0])  # horizons in years, Figure 3(b)
EXP4_STEPS_PER_YEAR_GRID_PAPER = (360, 720)  # simulation grids, Figure 3 caption
EXP4_N_PATHS_PAPER = 200_000  # paths per cell, Figure 3 caption
EXP4_N_BATCHES = 40  # batches of 5,000 paths, Figure 3 caption
EXP4_SEED = 3  # random seed, Figure 3 caption

EXP4_STEPS_PER_YEAR_GRID_RUN = (360, 720)  # grids actually executed in this replication
EXP4_N_PATHS_RUN = 200_000  # paths per cell actually executed

# ---------------------------------------------------------------------------
# Experiment 5: calibration to Bitcoin and Ether option smiles
# ---------------------------------------------------------------------------

EXP5_H_VALUES = (0.30, 0.50)  # fixed Hurst exponents calibrated, Sec. 5
EXP5_KAPPA2_MARGIN = 0.01  # numerical margin kappa_2 - beta >= margin (Sec. 5, Bitcoin fits)
# Bitcoin snapshot: Deribit quotes, 30 June 2023 08:00 UTC, maturities 7/28/91 days, 45 quotes
EXP5_BTC_MATURITIES_DAYS = (7, 28, 91)
EXP5_BTC_N_QUOTES = 45
# Ether snapshot: Deribit quotes, 26 October 2021 08:00 UTC, maturities 10/31/66 days, 39 quotes
EXP5_ETH_MATURITIES_DAYS = (10, 31, 66)
EXP5_ETH_N_QUOTES = 39

# Reported fitted parameter vectors, Table 4 (used only as a reference for comparison;
# they are not fed into the replication, which refits from data when data are available)
EXP5_TABLE4_REFERENCE = {
    "BTC_H0.30": dict(sigma0=0.36324, theta=0.36317, kappa1=3.80371, kappa2=0.19193,
                       beta=0.18193, eps=1.87830, rho=0.09641, kappa2_minus_beta=0.01000,
                       gamma=0.75098, rms=0.71651, inside=27, n_quotes=45),
    "BTC_H0.50": dict(sigma0=0.37746, theta=0.36609, kappa1=11.89875, kappa2=0.41383,
                       beta=0.40383, eps=4.09675, rho=0.09810, kappa2_minus_beta=0.01000,
                       gamma=0.71211, rms=0.79087, inside=28, n_quotes=45),
    "ETH_H0.30": dict(sigma0=0.77823, theta=3.92520, kappa1=0.01003, kappa2=0.57527,
                       beta=0.19698, eps=0.87174, rho=0.22041, kappa2_minus_beta=0.37828,
                       gamma=14.47392, rms=0.34923, inside=33, n_quotes=39),
    "ETH_H0.50": dict(sigma0=0.80466, theta=2.45560, kappa1=0.57978, kappa2=1.16610,
                       beta=0.42839, eps=1.87244, rho=0.22302, kappa2_minus_beta=0.73771,
                       gamma=3.18184, rms=0.30829, inside=33, n_quotes=39),
}

EXP5_SYNTHETIC_N_STRIKES = 11  # strikes per maturity for the synthetic self-test smile
EXP5_SYNTHETIC_SEED = 7  # random seed for the synthetic self-test
EXP5_SYNTHETIC_N_PATHS = 20000  # Monte Carlo paths for the synthetic self-test pricer
EXP5_SYNTHETIC_STEPS_PER_YEAR = 365  # simulation grid for the synthetic self-test pricer
