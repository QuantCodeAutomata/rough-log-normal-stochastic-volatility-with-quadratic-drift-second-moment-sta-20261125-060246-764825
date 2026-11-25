"""Exact linear-drift (kappa2=0) time-averaged second moment and the
Proposition 3.7 upper bound under quadratic drift, for the N-factor
Markovian-lifted kernel (paper Sec. 3.4, used by exp_4).

Custom implementation of the paper's formulas; scipy is used only for
quadrature and linear algebra.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from src.markovian_kernel import (
    KernelSpec, ell_N, resolvent_G_N_modes, G_N, resolvent_first_kind_L_over_T,
)


def exact_linear_drift_mean(t: np.ndarray, sigma0: float, theta: float, kappa1: float,
                             poles: np.ndarray, residues: np.ndarray) -> np.ndarray:
    """m(t) = theta + (sigma0-theta) E_alpha(-kappa1 t^alpha) becomes, for the
    Markovian kernel, m_N(t) = theta + (sigma0-theta) * R_N(t)/kappa1, where
    R_N is the resolvent of kappa1*K^N; equivalently m_N(t) solves the linear
    ODE system directly. Here we use the closed-form
    m_N(t) = theta + (sigma0-theta) * [1 - kappa1 * cumulative_int_0^t G_N(u) du]
    obtained from m_N(t) = sigma0 - kappa1 * int_0^t G_N(t-s)(m_N(s)-...)... ;
    for the deterministic linear case this reduces (Sec. 3.2) to
    m_N(t) = theta + (sigma0 - theta) * h_N(t), with h_N solving
    h_N(t) = 1 - kappa1 * int_0^t G_N(u) h_N(t-u) du, h_N(0) = 1.
    Because G_N is itself the resolvent of kappa1 K^N, h_N(t) = 1 - kappa1 * int_0^t G_N(u) du
    exactly when kappa1 K^N is used consistently (Volterra resolvent identity).
    """
    t = np.asarray(t, dtype=float)
    # int_0^t G_N(u) du = sum_k c_k/z_k (exp(z_k t) - 1)
    cum = np.tensordot((np.exp(np.multiply.outer(t, poles)) - 1.0) / poles[None, :],
                        residues, axes=([-1], [0]))
    h_N = 1.0 - kappa1 * np.real(cum)
    return theta + (sigma0 - theta) * h_N


def exact_linear_drift_second_moment(T_grid: np.ndarray, sigma0: float, theta: float,
                                      kappa1: float, kernel: KernelSpec, theta_vol: float,
                                      n_grid: int = 4000) -> np.ndarray:
    """E[I_T] = int_0^T v(t) dt for the linear-drift (kappa2=0) lifted model, where
    v solves the Volterra equation v(s) = m(s)^2 + theta_vol^2 int_0^s G_N(s-r)^2 v(r) dr
    (Sec. 3.2), via product integration on a uniform grid of `n_grid` steps per year.
    """
    poles, residues = resolvent_G_N_modes(kernel, kappa1)
    T_max = float(np.max(T_grid))
    n_steps = int(round(n_grid * T_max))
    dt = T_max / n_steps
    s_grid = np.arange(0, n_steps + 1) * dt

    m = exact_linear_drift_mean(s_grid, sigma0, theta, kappa1, poles, residues)
    G_vals = G_N(np.arange(0, n_steps + 1) * dt, poles, residues)
    kernel_h = theta_vol ** 2 * G_vals ** 2

    # Product integration (trapezoidal weights) for the Volterra equation
    # v(s) = m(s)^2 + int_0^s kernel_h(s-r) v(r) dr, solved forward in s with the
    # unknown v[i] isolated on the left-hand side (implicit trapezoidal weight at r=i).
    v = np.empty(n_steps + 1)
    v[0] = m[0] ** 2
    for i in range(1, n_steps + 1):
        r = np.arange(0, i)
        weights = np.full(i, dt)
        weights[0] *= 0.5
        explicit_sum = np.sum(kernel_h[i - r] * v[r] * weights)
        implicit_coef = 0.5 * dt * kernel_h[0]
        v[i] = (m[i] ** 2 + explicit_sum) / (1.0 - implicit_coef)

    cum_I = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * dt)])
    idx = np.round(T_grid / dt).astype(int)
    return cum_I[idx]


def proposition_3_7_bound(kappa1: float, kappa2: float, theta: float, sigma0: float,
                           kernel: KernelSpec, T: float | None = None) -> float:
    """Proposition 3.7 upper bound on A_T = E[I_T]/T.

    With T given, uses the finite-horizon bound (20): A_T <= y_T^2 with
        y_T = (c1+ + sqrt(c1+^2 + 4 kappa2 (kappa1 theta + sigma0 L([0,T])/T))) / (2 kappa2),
    where c1 = kappa2 theta - kappa1, c1+ = max(c1,0), and L([0,T])/T is the finite-T
    resolvent-of-the-first-kind ratio of K^N (Appendix A proof of Prop. 3.7).
    With T=None, uses the T->infinity limit (22), replacing L([0,T])/T by ell_N.
    Independent of theta_vol (volatility-of-volatility) and H in both cases.
    """
    if kappa2 <= 0:
        raise ValueError("Proposition 3.7 bound requires kappa2 > 0")
    c1 = kappa2 * theta - kappa1
    c1_plus = max(c1, 0.0)
    l_ratio = ell_N(kernel) if T is None else resolvent_first_kind_L_over_T(kernel, T)
    inside = c1_plus ** 2 + 4.0 * kappa2 * (kappa1 * theta + sigma0 * l_ratio)
    return ((c1_plus + np.sqrt(inside)) / (2.0 * kappa2)) ** 2
