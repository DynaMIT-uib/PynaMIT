"""Tests for the MAGE weighted-wind source algebra."""

from types import SimpleNamespace

import numpy as np

from pynamit.simulation.electrodynamics.ionospheric_closure import (
    electric_field_from_weighted_winds,
)


def _dummy_field(br, btheta, bphi, B0=4.7e-5):
    br = np.asarray(br, dtype=float)
    btheta = np.asarray(btheta, dtype=float)
    bphi = np.asarray(bphi, dtype=float)
    return SimpleNamespace(
        unit_br=br,
        unit_btheta=btheta,
        unit_bphi=bphi,
        Br=B0 * br,
        Btheta=B0 * btheta,
        Bphi=B0 * bphi,
    )


def _pynamit_resistance_tensor(SigmaP, SigmaH, br, btheta, bphi):
    etaP = SigmaP / (SigmaH**2 + SigmaP**2)
    etaH = SigmaH / (SigmaH**2 + SigmaP**2)
    b_p = np.array([[bphi**2 + br**2, -btheta * bphi], [-btheta * bphi, btheta**2 + br**2]])
    b_h = np.array([[0.0, br], [-br, 0.0]])
    return etaP * b_p + etaH * b_h


def _pynamit_resistance_values(SigmaP, SigmaH):
    denominator = SigmaH**2 + SigmaP**2
    return SigmaP / denominator, SigmaH / denominator


def _cross_spherical(a_r, a_theta, a_phi, b_r, b_theta, b_phi):
    """Test-side spherical-basis cross product."""
    return (
        a_theta * b_phi - a_phi * b_theta,
        a_phi * b_r - a_r * b_phi,
        a_r * b_theta - a_theta * b_r,
    )


def _weighted_wind_current_source(
    *, SigmaP, SigmaH, u_p_theta, u_p_phi, u_h_theta, u_h_phi, field
):
    """Return an independent reference for the 3D current source."""
    zero = np.zeros_like(u_p_theta)
    pedersen = np.asarray(
        _cross_spherical(zero, u_p_theta, u_p_phi, field.Br, field.Btheta, field.Bphi)
    )
    hall_wind = np.asarray(
        _cross_spherical(zero, u_h_theta, u_h_phi, field.Br, field.Btheta, field.Bphi)
    )
    hall = np.asarray(
        _cross_spherical(
            field.unit_br,
            field.unit_btheta,
            field.unit_bphi,
            hall_wind[0],
            hall_wind[1],
            hall_wind[2],
        )
    )
    return tuple(SigmaP * pedersen[index] + SigmaH * hall[index] for index in range(3))


def _q_eff_reference_for_pynamit(
    *,
    SigmaP,
    SigmaH,
    u_p_theta,
    u_p_phi,
    u_h_theta,
    u_h_phi,
    field,
    parallel_conductance=np.inf,
    br_floor=1e-3,
):
    """Return the Eq. A8 current proxy with PynaMIT's input sign."""
    b_r = np.asarray(field.unit_br, dtype=float).reshape(-1)
    b_theta = np.asarray(field.unit_btheta, dtype=float).reshape(-1)
    b_phi = np.asarray(field.unit_bphi, dtype=float).reshape(-1)
    SigmaP = np.asarray(SigmaP, dtype=float).reshape(-1)
    SigmaH = np.asarray(SigmaH, dtype=float).reshape(-1)
    q_r, q_theta, q_phi = _weighted_wind_current_source(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
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
        denominator = SigmaP * (b_theta**2 + b_phi**2) + sigma_parallel * b_r**2
        correction_theta = (
            ((sigma_parallel - SigmaP) * b_r * b_theta + SigmaH * b_phi) * q_r / denominator
        )
        correction_phi = (
            ((sigma_parallel - SigmaP) * b_r * b_phi - SigmaH * b_theta) * q_r / denominator
        )

    q_eff_theta_physical = q_theta - correction_theta
    q_eff_phi_physical = q_phi - correction_phi
    return -q_eff_theta_physical, -q_eff_phi_physical


def test_q_eff_reference_matches_appendix_a8_projection_with_pynamit_sign():
    """Test-side Q_eff reference uses Eq. A8 with PynaMIT's sign."""
    SigmaP = np.array([7.0, 5.0])
    SigmaH = np.array([2.5, 1.0])
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
        SigmaP=SigmaP,
        SigmaH=SigmaH,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
        parallel_conductance=sigma_parallel,
    )

    zero = np.zeros_like(u_p_theta)
    q_p = np.asarray(
        _cross_spherical(zero, u_p_theta, u_p_phi, field.Br, field.Btheta, field.Bphi)
    )
    q_h_wind = np.asarray(
        _cross_spherical(zero, u_h_theta, u_h_phi, field.Br, field.Btheta, field.Bphi)
    )
    q_h = np.asarray(_cross_spherical(br, btheta, bphi, q_h_wind[0], q_h_wind[1], q_h_wind[2]))
    q = SigmaP * q_p + SigmaH * q_h

    denominator = SigmaP * (btheta**2 + bphi**2) + sigma_parallel * br**2
    correction_theta = (
        ((sigma_parallel - SigmaP) * br * btheta + SigmaH * bphi) * q[0] / denominator
    )
    correction_phi = ((sigma_parallel - SigmaP) * br * bphi - SigmaH * btheta) * q[0] / denominator
    expected_physical_theta = q[1] - correction_theta
    expected_physical_phi = q[2] - correction_phi

    np.testing.assert_allclose(q_eff_theta, -expected_physical_theta)
    np.testing.assert_allclose(q_eff_phi, -expected_physical_phi)


def test_q_eff_matches_height_independent_wind_forcing():
    """Height-independent Q_eff equals wind forcing."""
    SigmaP = np.array([8.0, 4.0, 6.0])
    SigmaH = np.array([3.0, 1.5, 2.0])
    br = np.array([-0.91, -0.62, 0.73])
    btheta = np.array([0.22, -0.54, 0.12])
    bphi = np.array([0.35, 0.56, -0.67])
    norm = np.sqrt(br**2 + btheta**2 + bphi**2)
    br, btheta, bphi = br / norm, btheta / norm, bphi / norm
    field = _dummy_field(br, btheta, bphi)

    u_theta = np.array([75.0, -120.0, 40.0])
    u_phi = np.array([-35.0, 20.0, 95.0])
    q_eff_theta, q_eff_phi = _q_eff_reference_for_pynamit(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
        u_p_theta=u_theta,
        u_p_phi=u_phi,
        u_h_theta=u_theta,
        u_h_phi=u_phi,
        field=field,
        parallel_conductance=np.inf,
    )

    wind_cross_b = np.asarray(
        _cross_spherical(
            np.zeros_like(u_theta), u_theta, u_phi, field.Br, field.Btheta, field.Bphi
        )
    )
    q_eff_input = np.stack([q_eff_theta, q_eff_phi])
    for i in range(q_eff_input.shape[1]):
        resistance = _pynamit_resistance_tensor(SigmaP[i], SigmaH[i], br[i], btheta[i], bphi[i])
        E_from_q_eff = resistance @ q_eff_input[:, i]
        np.testing.assert_allclose(E_from_q_eff, -wind_cross_b[1:, i], rtol=1e-12, atol=1e-18)


def test_weighted_wind_electric_field_reduces_to_set_u_for_height_independent_wind():
    """Weighted-wind E should reduce to ordinary u forcing."""
    SigmaP = np.array([8.0, 4.0, 6.0])
    SigmaH = np.array([3.0, 1.5, 2.0])
    br = np.array([-0.91, -0.62, 0.73])
    btheta = np.array([0.22, -0.54, 0.12])
    bphi = np.array([0.35, 0.56, -0.67])
    norm = np.sqrt(br**2 + btheta**2 + bphi**2)
    br, btheta, bphi = br / norm, btheta / norm, bphi / norm
    field = _dummy_field(br, btheta, bphi)

    u_theta = np.array([75.0, -120.0, 40.0])
    u_phi = np.array([-35.0, 20.0, 95.0])
    etaP, etaH = _pynamit_resistance_values(SigmaP, SigmaH)

    e_neutral_wind_theta, e_neutral_wind_phi = electric_field_from_weighted_winds(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
        u_p_theta=u_theta,
        u_p_phi=u_phi,
        u_h_theta=u_theta,
        u_h_phi=u_phi,
        field=field,
        etaP=etaP,
        etaH=etaH,
    )
    wind_cross_b = np.asarray(
        _cross_spherical(
            np.zeros_like(u_theta), u_theta, u_phi, field.Br, field.Btheta, field.Bphi
        )
    )

    np.testing.assert_allclose(e_neutral_wind_theta, -wind_cross_b[1], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(e_neutral_wind_phi, -wind_cross_b[2], rtol=1e-12, atol=1e-18)


def test_weighted_wind_electric_field_matches_q_eff_away_from_equator():
    """Weighted-wind E and A8 Q_eff should agree off-equator."""
    SigmaP = np.array([8.0, 4.0, 6.0])
    SigmaH = np.array([3.0, 1.5, 2.0])
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
    etaP, etaH = _pynamit_resistance_values(SigmaP, SigmaH)

    e_neutral_wind_theta, e_neutral_wind_phi = electric_field_from_weighted_winds(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
        u_p_theta=u_p_theta,
        u_p_phi=u_p_phi,
        u_h_theta=u_h_theta,
        u_h_phi=u_h_phi,
        field=field,
        etaP=etaP,
        etaH=etaH,
    )
    q_eff_theta, q_eff_phi = _q_eff_reference_for_pynamit(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
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
        resistance = _pynamit_resistance_tensor(SigmaP[i], SigmaH[i], br[i], btheta[i], bphi[i])
        E_from_q_eff = resistance @ q_eff_input[:, i]
        np.testing.assert_allclose(
            E_from_q_eff,
            np.array([e_neutral_wind_theta[i], e_neutral_wind_phi[i]]),
            rtol=1e-12,
            atol=1e-18,
        )


def test_weighted_wind_electric_field_is_regular_at_dip_equator():
    """Equivalent E remains finite at the dip equator."""
    epsilon = 1e-9
    br = np.array([-epsilon, 0.0, epsilon])
    btheta = np.sqrt(1.0 - br**2)
    bphi = np.zeros_like(br)
    field = _dummy_field(br, btheta, bphi)
    SigmaP = np.full(3, 7.0)
    SigmaH = np.full(3, 2.5)
    etaP, etaH = _pynamit_resistance_values(SigmaP, SigmaH)

    e_theta, e_phi = electric_field_from_weighted_winds(
        SigmaP=SigmaP,
        SigmaH=SigmaH,
        u_p_theta=np.full(3, 80.0),
        u_p_phi=np.full(3, -35.0),
        u_h_theta=np.full(3, -20.0),
        u_h_phi=np.full(3, 110.0),
        field=field,
        etaP=etaP,
        etaH=etaH,
    )

    e_theta = np.asarray(e_theta)
    e_phi = np.asarray(e_phi)
    assert np.isfinite(e_theta).all()
    assert np.isfinite(e_phi).all()
    np.testing.assert_allclose(e_theta[[0, 2]], e_theta[1], rtol=0.0, atol=5e-12)
    np.testing.assert_allclose(e_phi[[0, 2]], e_phi[1], rtol=0.0, atol=5e-12)
