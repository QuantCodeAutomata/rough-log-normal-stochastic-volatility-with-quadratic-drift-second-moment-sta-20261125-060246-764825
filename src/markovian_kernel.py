"""Sum-of-exponentials (Markovian) approximation of the fractional kernel and
its exact resolvent, used for the N-factor lifted simulation engine and for
the exact linear-drift (kappa2=0) second moment of the lifted system
(paper Sec. 2.3, Sec. 3.4, Appendix B.1-B.4).

Custom implementation of the paper's Markovian lift (no general-purpose
library implements this rough-volatility kernel quadrature); numpy/scipy are
used only for polynomial root finding and symmetric eigendecomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, linalg


@dataclass(frozen=True)
class KernelSpec:
    """Nodes x_i (1/year) and weights w_i of K^N(t) = sum_i w_i exp(-x_i t)."""

    nodes: np.ndarray
    weights: np.ndarray

    @property
    def n_factors(self) -> int:
        return len(self.nodes)

    @property
    def wbar(self) -> float:
        """w_bar = sum_i w_i (Appendix B.1)."""
        return float(np.sum(self.weights))


def khat_N(z: complex, kernel: KernelSpec) -> complex:
    """Laplace transform K_hat^N(z) = sum_i w_i/(z+x_i) (Sec. 3.4)."""
    return np.sum(kernel.weights / (z + kernel.nodes))


def resolvent_G_N_modes(kernel: KernelSpec, kappa1: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact poles z_k and residues c_k of G_hat_N(z) = K_hat^N(z)/(1+kappa1 K_hat^N(z))
    (Sec. 3.4), so that G_N(t) = sum_k c_k exp(z_k t) for t >= 0.

    K_hat^N(z) = P(z)/Q0(z) with Q0(z) = prod_i (z+x_i) the common denominator and
    P(z) = sum_i w_i prod_{j!=i} (z+x_j). Hence
        G_hat_N(z) = P(z) / (Q0(z) + kappa1 P(z)),
    a rational function with the same numerator P and a perturbed denominator Q
    of the same degree N (poles z_k found as the roots of Q).
    """
    nodes, weights = kernel.nodes, kernel.weights
    N = kernel.n_factors
    Q0 = np.poly(-nodes)  # monic, degree N, roots at -x_i -> coefficients of prod(z+x_i)
    P = np.zeros(N)  # degree N-1 numerator, highest power first
    for i in range(N):
        other_roots = -np.delete(nodes, i)
        factor = np.poly(other_roots) if N > 1 else np.array([1.0])
        P += weights[i] * np.pad(factor, (N - len(factor), 0))
    Q = Q0 + kappa1 * np.pad(P, (1, 0))  # align P (degree N-1) with Q0 (degree N)
    poles = np.roots(Q)
    Qprime = np.polyder(Q)
    residues = np.polyval(P, poles) / np.polyval(Qprime, poles)
    return poles, residues


def G_N(t: np.ndarray, poles: np.ndarray, residues: np.ndarray) -> np.ndarray:
    """G_N(t) = sum_k c_k exp(z_k t), real-valued for real nodes/weights/kappa1."""
    t = np.asarray(t, dtype=float)
    out = np.zeros(t.shape, dtype=complex)
    for z_k, c_k in zip(poles, residues):
        out = out + c_k * np.exp(z_k * t[..., None]).reshape(t.shape) if t.ndim else out
    # vectorised form (works for any t shape)
    out = np.tensordot(np.exp(np.multiply.outer(t, poles)), residues, axes=([-1], [0]))
    return np.real(out)


def integral_G_N_squared(poles: np.ndarray, residues: np.ndarray) -> float:
    """int_0^inf G_N(u)^2 du = sum_{k,l} c_k c_l * (-1/(z_k+z_l)), closed form for a
    sum-of-exponentials resolvent with Re(z_k) < 0 (stable modes, Sec. 3.4)."""
    Z = poles[:, None] + poles[None, :]
    C = residues[:, None] * residues[None, :]
    value = -np.sum(C / Z)
    return float(np.real(value))


def integral_G_N_squared_quad(poles: np.ndarray, residues: np.ndarray, t_max: float = 50.0) -> float:
    """Independent cross-check of int_0^inf G_N(u)^2 du via direct quadrature."""
    def integrand(u):
        return G_N(np.array([u]), poles, residues)[0] ** 2

    value, _ = integrate.quad(integrand, 0.0, t_max, limit=200)
    return value


def ell_N(kernel: KernelSpec) -> float:
    """ell_N = (sum_i w_i/x_i)^{-1}, the Proposition 3.7 bound constant (Sec. 3.4).
    Requires all nodes x_i > 0 (true for the 3-factor and 97-factor kernels used here).
    """
    if np.any(kernel.nodes <= 0):
        raise ValueError("ell_N requires strictly positive nodes x_i")
    return 1.0 / float(np.sum(kernel.weights / kernel.nodes))


def build_97_factor_kernel(
    H: float = 0.30,
    t0: float = 1.0 / (365.0 * 512.0),
    cutoff_scale: float = 40.0,
    order: int = 8,
) -> KernelSpec:
    """Positive 97-factor Markovian kernel approximating the smoothed fractional
    kernel at H=0.30 used for the rough Bitcoin/Ether calibration (paper
    Appendix B.4, Eq. B9: "8 nodes on each of 12 intervals plus 1 exponential
    node").

    The exact interval layout and node placement are not given in the paper
    text; this reproduces the published construction of the companion code
    repository for this paper (ArturSepp/StochVolModels, paper replication
    script `rough_logsv_table5_kernel.py`, function `table5_kernel`): a
    two-mass regularisation of the Riemann-Liouville kernel singularity at
    t=0, discretised via Gauss-Jacobi quadrature (with the x^{power-1} weight)
    on the first interval and Gauss-Legendre quadrature on each subsequent
    interval of geometrically increasing length (factor 4), down-weighted by
    exp(-t0*x), plus a single exponential tail node carrying the remaining
    mass beyond the cutoff. The quadrature is validated in
    tests/test_markovian_kernel.py against the authors' published nodes and
    weights for H=0.30 (97 positive nodes, relative agreement < 1e-8).
    """
    from scipy.optimize import brentq
    from scipy.special import gamma as gamma_fn, gammaincc, roots_jacobi, roots_legendre

    alpha, power = H + 0.5, 0.5 - H
    normalizer = gamma_fn(alpha)

    def square_mass_error(log_rate: float) -> float:
        rate = np.exp(log_rate)
        cross = np.exp(rate + (1 - alpha) * log_rate) * normalizer * gammaincc(alpha, rate) / alpha
        return -1.0 / (2.0 * H) + 2.0 * cross + rate / (2.0 * alpha ** 2)

    rate = np.exp(brentq(square_mass_error, -700.0, np.log(100.0), xtol=1e-13))
    amplitude = rate / alpha
    constant = 1.0 / (gamma_fn(power) * normalizer)
    cutoff = cutoff_scale / t0
    left, right = 0.0, 1.0 / (91.0 / 365.0)
    nodes: list[float] = []
    weights: list[float] = []
    while left < cutoff:
        if left == 0.0:
            z, w = roots_jacobi(order, 0.0, power - 1.0)
            x = right * (z + 1.0) / 2.0
            w = w * (right / 2.0) ** power
        else:
            z, w = roots_legendre(order)
            x = left + (right - left) * (z + 1.0) / 2.0
            w = w * ((right - left) / 2.0) * x ** (power - 1.0)
        nodes.extend(x)
        weights.extend(constant * w * np.exp(-t0 * x))
        left, right = right, min(4.0 * right, cutoff)

    nodes.append(rate / t0)
    weights.append(amplitude * t0 ** (-power) / normalizer)
    return KernelSpec(nodes=np.asarray(nodes), weights=np.asarray(weights))


def build_quadrature_kernel(H: float, n_nodes: int, t_min: float, t_max: float) -> KernelSpec:
    """Sum-of-exponentials approximation of the Riemann-Liouville kernel
    K(t) = t^{H-1/2}/Gamma(H+1/2) for 0 < H < 1/2, via trapezoidal quadrature of
    its completely-monotone Laplace representation

        t^{H-1/2} = (1/Gamma(1/2-H)) int_0^inf x^{-H-1/2} e^{-x t} dx,

    discretized on a log-spaced grid x = exp(y) chosen so that x t spans a
    representative range over [t_min, t_max] (Appendix B.4 approximates the
    kernel by a similar finite sum of exponentials via Gaussian quadrature on
    several intervals of the Laplace domain; the exact 97-node placement of
    Appendix B.4, Eq. (B9), is not specified in the paper, so this generic,
    reusable quadrature is used here as a documented substitute, validated
    against the exact closed-form stability constants of Sec. 3.1-3.3 as
    n_nodes grows).
    """
    if not 0.0 < H < 0.5:
        raise ValueError("build_quadrature_kernel requires 0 < H < 0.5")
    from scipy.special import gamma as gamma_fn

    s = 0.5 - H
    C_H = 1.0 / (gamma_fn(H + 0.5) * gamma_fn(s))
    # Range wide enough that x*t is negligible at y_min and overwhelmingly large at
    # y_max for every t in [t_min, t_max] (the integrand e^{ys-e^y t} is then
    # vanishingly small at both ends, so the trapezoidal rule on a finite window
    # reproduces the improper integral to high accuracy).
    y_min, y_max = np.log(1.0e-5 / t_max), np.log(40.0 / t_min)
    y = np.linspace(y_min, y_max, n_nodes)
    dy = y[1] - y[0]
    nodes = np.exp(y)
    weights = C_H * dy * np.exp(y * s)
    # trapezoidal end-point correction
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return KernelSpec(nodes=nodes, weights=weights)


def build_symmetric_lambda(kernel: KernelSpec, c1_tilde: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigendecomposition of Lambda = diag(x) - c1_tilde * 1 w^T via conjugation with
    D = diag(sqrt(w)) (Appendix B.1): S = D Lambda D^{-1} = diag(x) - c1_tilde sqrt(w) sqrt(w)^T
    is symmetric, so S = Q diag(eigvals) Q^T with scipy.linalg.eigh.

    Returns (eigvals, Q, sqrt_w); Lambda = D^{-1} Q diag(eigvals) Q^T D.
    """
    sqrt_w = np.sqrt(kernel.weights)
    S = np.diag(kernel.nodes) - c1_tilde * np.outer(sqrt_w, sqrt_w)
    eigvals, Q = linalg.eigh(S)
    return eigvals, Q, sqrt_w
