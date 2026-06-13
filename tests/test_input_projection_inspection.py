"""Tests for projected-input inspection helpers."""

import numpy as np

import pynamit
from pynamit.visualization.input_projection import evaluate_projected_input


def test_evaluate_projected_scalar_input_on_model_grid(tmp_path):
    """Projected scalar inputs can be inspected on the model grid."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
    )
    coeffs = np.zeros(dynamics.input_field_spaces["jr"].coefficient_shape)
    coeffs[0] = 1.0
    dynamics.set_jr(coeffs, time=0.0, coefficients=True)

    values = evaluate_projected_input(dynamics, "jr", 0.0)

    assert set(values) == {"jr"}
    assert values["jr"].shape == (dynamics.state.geometry.grid.size,)
    assert np.all(np.isfinite(values["jr"]))


def test_evaluate_projected_conductance_returns_physical_conductance(tmp_path):
    """Conductance inspection includes SigmaP/SigmaH derived values."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
    )
    coeff_shape = dynamics.input_field_spaces["conductance"].coefficient_shape
    etaP = np.zeros(coeff_shape)
    etaH = np.zeros(coeff_shape)
    etaP[0] = 2.0
    etaH[0] = 1.0
    dynamics.set_resistance(etaP, etaH, time=0.0, coefficients=True)

    values = evaluate_projected_input(dynamics, "conductance", 0.0)

    assert {"etaP", "etaH", "SigmaP", "SigmaH"} <= set(values)
    np.testing.assert_allclose(
        values["SigmaP"],
        values["etaP"] / (values["etaP"] ** 2 + values["etaH"] ** 2),
    )
    np.testing.assert_allclose(
        values["SigmaH"],
        values["etaH"] / (values["etaP"] ** 2 + values["etaH"] ** 2),
    )


def test_evaluate_projected_tangential_input_returns_components(tmp_path):
    """Tangential inputs expose theta, phi, and magnitude maps."""
    dynamics = pynamit.Dynamics(
        run_directory=tmp_path,
        Nmax=2,
        Mmax=1,
        Ncs=8,
        ignore_PFAC=True,
    )
    coeff_length = dynamics.input_field_spaces["u"].index_length
    cf_coeffs = np.zeros(coeff_length)
    df_coeffs = np.zeros(coeff_length)
    cf_coeffs[0] = 1.0
    df_coeffs[0] = 0.5
    dynamics.set_neutral_wind(cf_coeffs, df_coeffs, time=0.0, coefficients=True)

    values = evaluate_projected_input(dynamics, "u", 0.0)

    assert {"u_theta", "u_phi", "u_mag"} <= set(values)
    np.testing.assert_allclose(
        values["u_mag"],
        np.sqrt(values["u_theta"] ** 2 + values["u_phi"] ** 2),
    )
