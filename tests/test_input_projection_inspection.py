"""Tests for projected-input inspection helpers."""

import numpy as np

import pynamit
from pynamit.visualization.input_projection import evaluate_projected_input


def test_evaluate_projected_scalar_input_on_model_grid(tmp_path):
    """Projected scalar inputs can be inspected on the model grid."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, enable_pfac_coupling=False
    )
    coeffs = np.zeros(simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape)
    coeffs[0] = 1.0
    simulation.set_jr(jr_coefficients=coeffs, time=0.0)

    values = evaluate_projected_input(simulation, "jr", 0.0)

    assert set(values) == {"jr"}
    assert values["jr"].shape == (simulation.geometry.model_grid.size,)
    assert np.all(np.isfinite(values["jr"]))


def test_evaluate_projected_input_corrects_explicit_transform_source(tmp_path):
    """Explicit target transforms keep the input source basis."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, enable_pfac_coupling=False
    )
    coeffs = np.zeros(simulation.run_data.schema.input_field_spaces["jr"].coefficient_shape)
    coeffs[0] = 1.0
    simulation.set_jr(jr_coefficients=coeffs, time=0.0)

    grid = simulation.geometry.model_grid
    wrong_source_transform = pynamit.SphericalTransform(simulation.run_data.schema.sh_basis, grid)

    corrected = evaluate_projected_input(simulation, "jr", 0.0, transform=wrong_source_transform)
    default = evaluate_projected_input(simulation, "jr", 0.0, grid=grid)

    np.testing.assert_allclose(corrected["jr"], default["jr"])


def test_evaluate_projected_conductance_returns_physical_conductance(tmp_path):
    """Conductance inspection includes SigmaP/SigmaH derived values."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, enable_pfac_coupling=False
    )
    coeff_shape = simulation.run_data.schema.input_field_spaces["resistance"].coefficient_shape
    etaP = np.zeros(coeff_shape)
    etaH = np.zeros(coeff_shape)
    etaP[0] = 2.0
    etaH[0] = 1.0
    simulation.set_resistance(etaP_coefficients=etaP, etaH_coefficients=etaH, time=0.0)

    values = evaluate_projected_input(simulation, "resistance", 0.0)

    assert {"etaP", "etaH", "SigmaP", "SigmaH"} <= set(values)
    np.testing.assert_allclose(
        values["SigmaP"], values["etaP"] / (values["etaP"] ** 2 + values["etaH"] ** 2)
    )
    np.testing.assert_allclose(
        values["SigmaH"], values["etaH"] / (values["etaP"] ** 2 + values["etaH"] ** 2)
    )


def test_evaluate_projected_tangential_input_returns_components(tmp_path):
    """Tangential inputs expose theta, phi, and magnitude maps."""
    simulation = pynamit.Simulation(
        run_directory=tmp_path, Nmax=2, Mmax=1, Ncs=8, enable_pfac_coupling=False
    )
    coeff_length = simulation.run_data.schema.input_field_spaces["u"].index_length
    cf_coeffs = np.zeros(coeff_length)
    df_coeffs = np.zeros(coeff_length)
    cf_coeffs[0] = 1.0
    df_coeffs[0] = 0.5
    simulation.set_neutral_wind(u_cf=cf_coeffs, u_df=df_coeffs, time=0.0)

    values = evaluate_projected_input(simulation, "u", 0.0)

    assert {"u_theta", "u_phi", "u_mag"} <= set(values)
    np.testing.assert_allclose(
        values["u_mag"], np.sqrt(values["u_theta"] ** 2 + values["u_phi"] ** 2)
    )
