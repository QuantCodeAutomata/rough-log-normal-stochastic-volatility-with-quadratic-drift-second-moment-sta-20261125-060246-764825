"""Symmetric-splitting Monte Carlo scheme for the Markovian-lifted log-normal
Volterra stochastic-volatility model with quadratic drift (paper Appendix B,
Eq. B1-B5). This module is the shared simulation engine used by exp_2, exp_3,
exp_4 and exp_5.

Custom implementation of the paper's own splitting scheme (B1)-(B5); no
general-purpose library implements this model. numpy is used only for array
operations and random-number generation; scipy.linalg is used for the
symmetric eigendecomposition of the memory-map generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.markovian_kernel import KernelSpec, build_symmetric_lambda


class SimulationError(RuntimeError):
    """Raised when the simulated volatility state becomes non-positive or non-finite.

    Per the paper's error-handling convention (Appendix B.4): no floor, no reset,
    no deletion of paths -- a non-positive or non-finite state stops the run.
    """


@dataclass(frozen=True)
class ModelParams:
    """Risk-neutral parameters of the log-normal Volterra model with quadratic
    drift (paper Sec. 2, Eq. (1)-(2))."""

    sigma0: float  # initial instantaneous volatility
    kappa1: float  # linear mean-reversion speed (1/year)
    kappa2: float  # quadratic damping speed (1/(year x vol))
    theta: float  # mean-reversion target level (drift), kappa1*theta, kappa2*theta
    beta: float  # spot-volatility covariance loading
    eps: float  # idiosyncratic volatility-of-volatility loading

    @property
    def theta_vol(self) -> float:
        """theta_total = sqrt(beta^2+eps^2), the aggregate volatility-of-volatility
        driving the common Brownian motion W^sigma (Appendix B.1)."""
        return float(np.sqrt(self.beta ** 2 + self.eps ** 2))

    @property
    def rho(self) -> float:
        """rho = beta/theta_total, the spot-volatility correlation (Sec. 2)."""
        return self.beta / self.theta_vol


@dataclass(frozen=True)
class SplittingOperators:
    """Precomputed, parameter-dependent pieces of the symmetric splitting scheme
    (Appendix B.1-B.3), built once per (kernel, params) pair and reused across
    every simulated step and path."""

    kernel: KernelSpec
    params: ModelParams
    c0: float
    c1: float
    c1_tilde: float
    eigvals: np.ndarray
    Q: np.ndarray
    sqrt_w: np.ndarray
    sigma0_i: np.ndarray  # sigma_{0,i} = sigma0/wbar, per-factor reference level

    @classmethod
    def build(cls, kernel: KernelSpec, params: ModelParams) -> "SplittingOperators":
        wbar = kernel.wbar
        c0 = params.kappa1 * params.theta
        c1 = params.kappa2 * params.theta - params.kappa1
        c1_tilde = c1 - 0.5 * wbar * params.theta_vol ** 2
        eigvals, Q, sqrt_w = build_symmetric_lambda(kernel, c1_tilde)
        if np.any(eigvals <= 0):
            raise SimulationError(
                "Memory-map generator Lambda is not positive definite for these "
                "parameters (eigenvalues must be > 0); reduce kappa2 or theta_vol."
            )
        sigma0_i = np.full(kernel.n_factors, params.sigma0 / wbar)
        return cls(kernel=kernel, params=params, c0=c0, c1=c1, c1_tilde=c1_tilde,
                   eigvals=eigvals, Q=Q, sqrt_w=sqrt_w, sigma0_i=sigma0_i)

    def exp_minus_lambda(self, tau: float) -> np.ndarray:
        """exp(-Lambda*tau) = D^{-1} Q diag(exp(-eigvals tau)) Q^T D (Appendix B.2)."""
        D = self.sqrt_w
        core = (self.Q * np.exp(-self.eigvals * tau)) @ self.Q.T
        return (core / D[:, None]) * D[None, :]

    def u_star(self) -> np.ndarray:
        """U* = (c0 + c1_tilde*sigma0) * Lambda^{-1} 1 (Appendix B.2)."""
        D = self.sqrt_w
        inv_eig = 1.0 / self.eigvals
        lambda_inv_ones = (self.Q @ (inv_eig * (self.Q.T @ D))) / D
        return (self.c0 + self.c1_tilde * self.params.sigma0) * lambda_inv_ones


def memory_map(U: np.ndarray, ops: SplittingOperators, tau: float) -> np.ndarray:
    """M_tau (Appendix B.2): U -> U* + exp(-Lambda tau)(U - U*)."""
    u_star = ops.u_star()
    expL = ops.exp_minus_lambda(tau)
    return u_star[:, None] + expL @ (U - u_star[:, None])


def damping_map(U: np.ndarray, ops: SplittingOperators, tau: float) -> np.ndarray:
    """D_tau (Appendix B.3): sigma^N -> sigma^N/(1+wbar kappa2 tau sigma^N); each
    factor receives the same additive increment (Delta sigma^N)/wbar."""
    kappa2, wbar = ops.params.kappa2, ops.kernel.wbar
    sigma_N = aggregate_sigma(U, ops)
    if kappa2 == 0.0:
        return U
    sigma_N_new = sigma_N / (1.0 + wbar * kappa2 * tau * sigma_N)
    delta = (sigma_N_new - sigma_N) / wbar
    return U + delta[None, :]


def noise_map(U: np.ndarray, ops: SplittingOperators, h: float, Z: np.ndarray) -> np.ndarray:
    """N_h (Appendix B.3): sigma^N -> sigma^N exp(wbar theta_vol sqrt(h) Z); each
    factor receives the same additive increment (Delta sigma^N)/wbar."""
    wbar, theta_vol = ops.kernel.wbar, ops.params.theta_vol
    sigma_N = aggregate_sigma(U, ops)
    sigma_N_new = sigma_N * np.exp(wbar * theta_vol * np.sqrt(h) * Z)
    delta = (sigma_N_new - sigma_N) / wbar
    return U + delta[None, :]


def aggregate_sigma(U: np.ndarray, ops: SplittingOperators) -> np.ndarray:
    """sigma^N = sigma0 + w^T U (Sec. 2.3)."""
    return ops.params.sigma0 + ops.kernel.weights @ U


def split_step(U: np.ndarray, ops: SplittingOperators, h: float, Z: np.ndarray) -> np.ndarray:
    """One symmetric-splitting step (Eq. B4):
    U_{n+1} = M_{h/2} . D_{h/2} . N_h . D_{h/2} . M_{h/2} (U_n)."""
    U = memory_map(U, ops, h / 2.0)
    U = damping_map(U, ops, h / 2.0)
    U = noise_map(U, ops, h, Z)
    U = damping_map(U, ops, h / 2.0)
    U = memory_map(U, ops, h / 2.0)
    return U


def simulate_paths(
    kernel: KernelSpec,
    params: ModelParams,
    T: float,
    steps_per_year: int,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
) -> dict:
    """Simulate `n_paths` paths of the Markovian-lifted model on [0, T] using the
    symmetric splitting scheme (Eq. B4) and the price step (Eq. B5).

    Returns a dict with keys:
        'I'        : (n_paths,) terminal quadratic variation sum (sigma_n^N)^2 h
        'J'        : (n_paths,) terminal martingale sum sigma_n^N sqrt(h) Z_n^sigma
        'X'        : (n_paths,) terminal log-price increment X_n - X_0
        'rho'      : spot-volatility correlation used
        'theta_vol': total volatility-of-volatility used
        'n_steps'  : number of simulated steps
        'h'        : step size (years)
    Raises SimulationError if the volatility state becomes non-positive or
    non-finite at any step (no floor, no reset, no deletion of paths).
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when antithetic=True")

    ops = SplittingOperators.build(kernel, params)
    N_steps = int(round(T * steps_per_year))
    h = T / N_steps
    rho, theta_vol = params.rho, params.theta_vol

    rng = np.random.default_rng(seed)
    n_half = n_paths // 2 if antithetic else n_paths

    U = np.zeros((kernel.n_factors, n_paths))
    I = np.zeros(n_paths)
    J = np.zeros(n_paths)
    X = np.zeros(n_paths)

    for n in range(N_steps):
        z_half = rng.standard_normal((2, n_half))
        if antithetic:
            Z_sigma = np.concatenate([z_half[0], -z_half[0]])
            Z_perp = np.concatenate([z_half[1], -z_half[1]])
        else:
            Z_sigma, Z_perp = z_half[0], z_half[1]

        sigma_N_begin = aggregate_sigma(U, ops)
        if not np.all(np.isfinite(sigma_N_begin)) or np.any(sigma_N_begin <= 0.0):
            raise SimulationError(
                f"Non-positive or non-finite volatility at step {n}/{N_steps}"
            )

        sqrt_h = np.sqrt(h)
        I += sigma_N_begin ** 2 * h
        J += sigma_N_begin * sqrt_h * Z_sigma
        X += (-0.5 * sigma_N_begin ** 2 * h
              + sigma_N_begin * sqrt_h * (rho * Z_sigma + np.sqrt(1.0 - rho ** 2) * Z_perp))

        U = split_step(U, ops, h, Z_sigma)

    return {"I": I, "J": J, "X": X, "rho": rho, "theta_vol": theta_vol,
            "n_steps": N_steps, "h": h}
