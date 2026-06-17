"""Tests for the MAGE weighted-wind source algebra."""

from types import SimpleNamespace

import numpy as np

from scripts.simulation.mage_forcing_final import (
    cross_spherical,
    direct_E_source_for_pynamit,
    weighted_wind_current_source,
)


def _dummy_field(br, btheta, bphi, B0=4.7e-5):
    br = np.asarray(br, dtype=float)
    btheta = np.asarray(btheta, dtype=float)
    bphi = np.asarray(bphi, dtype=float)
    return SimpleNamespace(
        br=br, btheta=btheta, bphi=bphi, Br=B0 * br, Btheta=B0 * btheta, Bphi=B0 * bphi
    )


def _pynamit_resistance_tensor(sigma_p, sigma_h, br, btheta, bphi):
    eta_p = sigma_p / (sigma_h**2 + sigma_p**2)
    eta_h = sigma_h / (sigma_h**2 + sigma_p**2)
    b_p = np.array([[bphi**2 + br**2, -btheta * bphi], [-btheta * bphi, btheta**2 + br**2]])
    b_h = np.array([[0.0, br], [-br, 0.0]])
    return eta_p * b_p + eta_h * b_h


def _pynamit_resistance_values(sigma_p, sigma_h):
    denominator = sigma_h**2 + sigma_p**2
    return sigma_p / denominator, sigma_h / denominator


def _q_eff_reference_for_pynamit(
    *,
    sigma_p,
    sigma_h,
    u_p_theta,
    u_p_phi,
    u_h_theta,
    u_h_phi,
    field,
    parallel_conductance=np.inf,
    br_floor=1e-3,
):
    """Return the Eq. A8 current proxy with PynaMIT's input sign."""
    b_r = np.asarray(field.br, dtype=float).reshape(-1)
    b_theta = np.asarray(field.btheta, dtype=float).reshape(-1)
    b_phi = np.asarray(field.bphi, dtype=float).reshape(-1)
    sigma_p = np.asarray(sigma_p, dtype=float).reshape(-1)
    sigma_h = np.asarray(sigma_h, dtype=float).reshape(-1)
    q_r, q_theta, q_phi = weighted_wind_current_source(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
    )

    if np.isinf(parallel_conductance):
        valid = np.abs(b_r) > br_floor
        correction_theta = np.divide(q_r * b_theta, b_r, out=np.zeros_like(q_r), where=valid)
        correction_phi = np.divide(q_r * b_phi, b_r, out=np.zeros_like(q_r), where=valid)
    else:
        sigma_parallel = float(parallel_conductance)
        denominator = sigma_p * (b_theta**2 + b_phi**2) + sigma_parallel * b_r**2
        correction_theta = (
            ((sigma_parallel - sigma_p) * b_r * b_theta + sigma_h * b_phi) * q_r / denominator
        )
        correction_phi = (
            ((sigma_parallel - sigma_p) * b_r * b_phi - sigma_h * b_theta) * q_r / denominator
        )

    q_eff_theta_physical = q_theta - correction_theta
    q_eff_phi_physical = q_phi - correction_phi
    return -q_eff_theta_physical, -q_eff_phi_physical


def test_q_eff_reference_matches_appendix_a8_projection_with_pynamit_sign():
    """Test-side Q_eff reference uses Eq. A8 with PynaMIT's sign."""
    sigma_p = np.array([7.0, 5.0])
    sigma_h = np.array([2.5, 1.0])
    sigma_parallel = 80.0
    br = np.array([-0.82, 0.74])
    btheta = np.array([0.31, -0.42])
    bphi = np.array([0.48, 0.52])
    norm = np.sqrt(br**2 + btheta**2 + bphi**2)
    br, btheta, bphi = br / norm, btheta / norm, bphi / norm
    field = _dummy_field(br, btheta, bphi)

    u_p_theta = np.array([140.0, -90.0])
    u_p_phi = np.array([-40.0, 60.0])
    u_h_theta = np.array([20.0, 130.0])
    u_h_phi = np.array([100.0, -70.0])

    q_eff_theta, q_eff_phi = _q_eff_reference_for_pynamit(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
        parallel_conductance=sigma_parallel,
    )

    zero = np.zeros_like(u_p_theta)
    q_p = np.asarray(cross_spherical(zero, u_p_theta, u_p_phi, field.Br, field.Btheta, field.Bphi))
    q_h_wind = np.asarray(
        cross_spherical(zero, u_h_theta, u_h_phi, field.Br, field.Btheta, field.Bphi)
    )
    q_h = np.asarray(cross_spherical(br, btheta, bphi, q_h_wind[0], q_h_wind[1], q_h_wind[2]))
    q = sigma_p * q_p + sigma_h * q_h

    denominator = sigma_p * (btheta**2 + bphi**2) + sigma_parallel * br**2
    correction_theta = (
        ((sigma_parallel - sigma_p) * br * btheta + sigma_h * bphi) * q[0] / denominator
    )
    correction_phi = (
        ((sigma_parallel - sigma_p) * br * bphi - sigma_h * btheta) * q[0] / denominator
    )
    expected_physical_theta = q[1] - correction_theta
    expected_physical_phi = q[2] - correction_phi

    np.testing.assert_allclose(q_eff_theta, -expected_physical_theta)
    np.testing.assert_allclose(q_eff_phi, -expected_physical_phi)


def test_q_eff_reduces_to_direct_neutral_wind_for_height_independent_wind():
    """Height-independent Q_eff equals direct wind forcing."""
    sigma_p = np.array([8.0, 4.0, 6.0])
    sigma_h = np.array([3.0, 1.5, 2.0])
    br = np.array([-0.91, -0.62, 0.73])
    btheta = np.array([0.22, -0.54, 0.12])
    bphi = np.array([0.35, 0.56, -0.67])
    norm = np.sqrt(br**2 + btheta**2 + bphi**2)
    br, btheta, bphi = br / norm, btheta / norm, bphi / norm
    field = _dummy_field(br, btheta, bphi)

    u_theta = np.array([75.0, -120.0, 40.0])
    u_phi = np.array([-35.0, 20.0, 95.0])
    q_eff_theta, q_eff_phi = _q_eff_reference_for_pynamit(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_theta,
        u_p_phi=u_phi,
        u_h_theta=u_theta,
        u_h_phi=u_phi,
        field=field,
        parallel_conductance=np.inf,
    )

    wind_cross_b = np.asarray(
        cross_spherical(np.zeros_like(u_theta), u_theta, u_phi, field.Br, field.Btheta, field.Bphi)
    )
    q_eff_input = np.stack([q_eff_theta, q_eff_phi])
    for i in range(q_eff_input.shape[1]):
        resistance = _pynamit_resistance_tensor(sigma_p[i], sigma_h[i], br[i], btheta[i], bphi[i])
        E_from_q_eff = resistance @ q_eff_input[:, i]
        np.testing.assert_allclose(E_from_q_eff, -wind_cross_b[1:, i], rtol=1e-12, atol=1e-18)


def test_direct_E_reduces_to_set_u_for_height_independent_wind():
    """Direct E weighted winds should reduce to ordinary u forcing."""
    sigma_p = np.array([8.0, 4.0, 6.0])
    sigma_h = np.array([3.0, 1.5, 2.0])
    br = np.array([-0.91, -0.62, 0.73])
    btheta = np.array([0.22, -0.54, 0.12])
    bphi = np.array([0.35, 0.56, -0.67])
    norm = np.sqrt(br**2 + btheta**2 + bphi**2)
    br, btheta, bphi = br / norm, btheta / norm, bphi / norm
    field = _dummy_field(br, btheta, bphi)

    u_theta = np.array([75.0, -120.0, 40.0])
    u_phi = np.array([-35.0, 20.0, 95.0])
    eta_p, eta_h = _pynamit_resistance_values(sigma_p, sigma_h)

    e_direct_theta, e_direct_phi = direct_E_source_for_pynamit(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_theta,
        u_p_phi=u_phi,
        u_h_theta=u_theta,
        u_h_phi=u_phi,
        field=field,
        eta_p=eta_p,
        eta_h=eta_h,
    )
    wind_cross_b = np.asarray(
        cross_spherical(np.zeros_like(u_theta), u_theta, u_phi, field.Br, field.Btheta, field.Bphi)
    )

    np.testing.assert_allclose(e_direct_theta, -wind_cross_b[1], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(e_direct_phi, -wind_cross_b[2], rtol=1e-12, atol=1e-18)


def test_direct_E_matches_q_eff_electric_field_away_from_equator():
    """Direct E and A8 Q_eff should match away from the equator."""
    sigma_p = np.array([8.0, 4.0, 6.0])
    sigma_h = np.array([3.0, 1.5, 2.0])
    br = np.array([-0.91, -0.62, 0.73])
    btheta = np.array([0.22, -0.54, 0.12])
    bphi = np.array([0.35, 0.56, -0.67])
    norm = np.sqrt(br**2 + btheta**2 + bphi**2)
    br, btheta, bphi = br / norm, btheta / norm, bphi / norm
    field = _dummy_field(br, btheta, bphi)

    u_p_theta = np.array([75.0, -120.0, 40.0])
    u_p_phi = np.array([-35.0, 20.0, 95.0])
    u_h_theta = np.array([20.0, 50.0, -85.0])
    u_h_phi = np.array([100.0, -30.0, 15.0])
    eta_p, eta_h = _pynamit_resistance_values(sigma_p, sigma_h)

    e_direct_theta, e_direct_phi = direct_E_source_for_pynamit(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
        eta_p=eta_p,
        eta_h=eta_h,
    )
    q_eff_theta, q_eff_phi = _q_eff_reference_for_pynamit(
        sigma_p=sigma_p,
        sigma_h=sigma_h,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
        parallel_conductance=np.inf,
        br_floor=1e-6,
    )

    q_eff_input = np.stack([q_eff_theta, q_eff_phi])
    for i in range(q_eff_input.shape[1]):
        resistance = _pynamit_resistance_tensor(sigma_p[i], sigma_h[i], br[i], btheta[i], bphi[i])
        E_from_q_eff = resistance @ q_eff_input[:, i]
        np.testing.assert_allclose(
            E_from_q_eff, np.array([e_direct_theta[i], e_direct_phi[i]]), rtol=1e-12, atol=1e-18
        )
