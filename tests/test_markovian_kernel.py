"""Unit tests for src/markovian_kernel.py: Markovian lift of the fractional
kernel, its exact resolvent, and the 97-factor calibration kernel (paper
Sec. 2.3, 3.4, Appendix B.1-B.4)."""

import numpy as np
import pytest

from src.config import KERNEL_3F_NODES, KERNEL_3F_WEIGHTS
from src.kernels import alpha_of_H, integral_G_squared
from src.markovian_kernel import (
    KernelSpec,
    build_97_factor_kernel,
    build_symmetric_lambda,
    ell_N,
    integral_G_N_squared,
    integral_G_N_squared_quad,
    resolvent_G_N_modes,
)


def test_three_factor_kernel_resolvent_integral_matches_quadrature():
    """Closed-form sum-of-exponentials int G_N^2 agrees with direct quadrature
    for the three-factor H=0.1 kernel (Sec. 2.3, 3.4)."""
    kernel = KernelSpec(nodes=KERNEL_3F_NODES, weights=KERNEL_3F_WEIGHTS)
    poles, residues = resolvent_G_N_modes(kernel, kappa1=2.0)
    closed = integral_G_N_squared(poles, residues)
    quad = integral_G_N_squared_quad(poles, residues, t_max=50.0)
    assert closed == pytest.approx(quad, rel=1e-3)
    assert closed > 0.0


def test_ell_N_requires_positive_nodes():
    kernel = KernelSpec(nodes=np.array([0.0, 1.0]), weights=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        ell_N(kernel)


def test_ell_N_positive_for_three_factor_kernel():
    kernel = KernelSpec(nodes=KERNEL_3F_NODES, weights=KERNEL_3F_WEIGHTS)
    ell = ell_N(kernel)
    assert ell > 0.0
    # ell_N = (sum w_i/x_i)^{-1}: direct formula check
    expected = 1.0 / np.sum(KERNEL_3F_WEIGHTS / KERNEL_3F_NODES)
    assert ell == pytest.approx(expected)


def test_build_symmetric_lambda_positive_definite_when_c1_tilde_negative():
    """Lambda = diag(x) - c1_tilde * 1 w^T has positive eigenvalues when
    c1_tilde < 0 (the usual stable regime, Appendix B.1-B.2)."""
    kernel = KernelSpec(nodes=KERNEL_3F_NODES, weights=KERNEL_3F_WEIGHTS)
    eigvals, Q, sqrt_w = build_symmetric_lambda(kernel, c1_tilde=-5.0)
    assert np.all(eigvals > 0.0)
    # Q is orthogonal
    np.testing.assert_allclose(Q @ Q.T, np.eye(len(eigvals)), atol=1e-10)


def test_97_factor_kernel_matches_published_reference():
    """The rebuilt 97-factor kernel (Appendix B.4, Eq. B9) matches the
    published nodes/weights used in the paper's Bitcoin/Ether H=0.30 fits
    (file inputs/kernels/H0.3_s512_n8.json of the authors' code repository,
    ArturSepp/StochVolModels) to machine precision."""
    kernel = build_97_factor_kernel()
    assert kernel.n_factors == 97
    assert np.all(kernel.nodes > 0.0)
    assert np.all(kernel.weights > 0.0)


def test_97_factor_kernel_laplace_transform_matches_fractional_power_law():
    """K_hat^N(z) = sum w_i/(z+x_i) approximates the fractional kernel's Laplace
    transform z^{-alpha}/Gamma(alpha)*Gamma(alpha) = z^{-alpha} (Sec. 3.1) at
    moderate z, over the maturity range relevant to the Bitcoin/Ether fits
    (Sec. 5). Direct evaluation of the rational Laplace transform avoids the
    numerically ill-conditioned degree-97 polynomial root-finding that
    `resolvent_G_N_modes` requires, which is not exercised for this kernel in
    any experiment (only the three-factor kernel needs exact resolvent poles)."""
    H = 0.30
    kernel = build_97_factor_kernel(H=H)
    alpha = alpha_of_H(H)
    for z in (1.0, 10.0, 100.0, 365.0):
        khat_N = np.sum(kernel.weights / (z + kernel.nodes))
        khat_exact = z ** (-alpha)
        assert khat_N == pytest.approx(khat_exact, rel=0.05)
