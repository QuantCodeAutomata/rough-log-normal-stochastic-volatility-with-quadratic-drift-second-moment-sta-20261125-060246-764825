"""Unit tests for src/kernels.py: stability parameter, theta_c, conditional
moments and the growth rate for the fractional (Riemann-Liouville) kernel."""

import numpy as np
import pytest

from src.kernels import (
    alpha_of_H,
    C_alpha,
    integral_G_squared,
    integral_G_squared_quad,
    theta_c,
    gamma_param,
    innovation_proxy,
    ResolventTable,
    conditional_second_moment,
    growth_rate_lambda_star,
    mittag_leffler_growth_curve,
)


def test_C_alpha_classical_limit():
    """C(1) = 1/2 at H=1/2 (Sec. 3.3)."""
    assert C_alpha(1.0) == pytest.approx(0.5)


def test_integral_G_squared_matches_closed_form_and_quadrature():
    """The closed-form spectral formula agrees with independent quadrature (Sec. 3.3)."""
    for H, k1 in [(0.1, 1.0), (0.1, 3.0), (0.3, 2.0), (0.5, 1.0), (0.5, 5.0)]:
        alpha = alpha_of_H(H)
        closed = integral_G_squared(alpha, k1)
        quad = integral_G_squared_quad(alpha, k1)
        assert closed == pytest.approx(quad, rel=1e-4)


@pytest.mark.parametrize(
    "H,kappa1,expected_theta_c,expected_nu",
    [
        (0.1, 1.0, 0.846, 0.730),
        (0.1, 3.0, 1.016, 0.877),
        (0.5, 1.0, 1.414, 0.089),
    ],
)
def test_table2_reference_values(H, kappa1, expected_theta_c, expected_nu):
    """Reproduces selected Table 2 entries (paper Sec. 3.3, Table 2, page 29)."""
    alpha = alpha_of_H(H)
    tc = theta_c(alpha, kappa1)
    nu = innovation_proxy(tc, H, 1.0 / 252.0)
    assert tc == pytest.approx(expected_theta_c, abs=2e-3)
    assert nu == pytest.approx(expected_nu, abs=2e-3)


def test_gamma_param_scaling():
    """gamma = theta^2 * int G^2 is increasing in theta and zero at theta=0."""
    alpha = alpha_of_H(0.3)
    assert gamma_param(0.0, alpha, 2.0) == 0.0
    g1 = gamma_param(1.0, alpha, 2.0)
    g2 = gamma_param(2.0, alpha, 2.0)
    assert g2 > g1 > 0.0


def test_gamma_param_kappa1_zero_is_infinite():
    """Without mean reversion (kappa1=0) the stability integral diverges (Sec. 3.3)."""
    alpha = alpha_of_H(0.2)
    assert np.isinf(gamma_param(1.0, alpha, 0.0))


def test_conditional_second_moment_subcritical_converges_to_limit():
    """For gamma < 1, E[sigma_t^2]/theta^2 -> 1/(1-gamma) as t -> infinity (Sec. 3.3)."""
    H = 0.3
    alpha = alpha_of_H(H)
    kappa1 = 2.0
    target_gamma = 0.5
    theta = np.sqrt(target_gamma / integral_G_squared(alpha, kappa1))
    table = ResolventTable.build(alpha, kappa1, t_max=8.0, n_points=150)
    t_grid, m_grid, v = conditional_second_moment(table, sigma0=theta, theta=theta, T=8.0, steps_per_year=200)
    ratio = v / theta ** 2
    expected_limit = 1.0 / (1.0 - target_gamma)
    assert ratio[-1] == pytest.approx(expected_limit, rel=0.1)
    assert ratio[0] == pytest.approx(1.0, abs=1e-8)


def test_conditional_second_moment_supercritical_grows():
    """For gamma > 1, the second moment grows without bound (Sec. 3.3)."""
    H = 0.3
    alpha = alpha_of_H(H)
    kappa1 = 2.0
    target_gamma = 1.2
    theta = np.sqrt(target_gamma / integral_G_squared(alpha, kappa1))
    table = ResolventTable.build(alpha, kappa1, t_max=5.0, n_points=150)
    _, _, v = conditional_second_moment(table, sigma0=theta, theta=theta, T=5.0, steps_per_year=200)
    assert v[-1] > v[len(v) // 2] > v[0]


def test_growth_rate_matches_paper_reference():
    """lambda* at H=0.1, theta=1.85: kappa1=0 gives ~17851/yr, kappa1=3 gives ~17235/yr (Sec. 3.3)."""
    H = 0.1
    alpha = alpha_of_H(H)
    theta = 1.85
    lam0 = growth_rate_lambda_star(theta, alpha, 0.0)
    lam3 = growth_rate_lambda_star(theta, alpha, 3.0)
    assert lam0 == pytest.approx(17851.0, rel=0.01)
    assert lam3 == pytest.approx(17235.0, rel=0.01)
    assert lam0 > lam3  # mean reversion moderately lowers the growth rate


def test_mittag_leffler_growth_matches_classical_case_H_half():
    """At H=1/2, E[sigma_t^2] = sigma0^2 exp(theta^2 t) exactly (Corollary 3.3)."""
    t = np.linspace(0, 2, 20)
    sigma0, theta = 0.3, 0.3
    growth = mittag_leffler_growth_curve(t, H=0.5, sigma0=sigma0, theta=theta)
    expected = sigma0 ** 2 * np.exp(theta ** 2 * t)
    np.testing.assert_allclose(growth, expected, rtol=1e-6)
