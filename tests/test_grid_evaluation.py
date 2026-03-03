"""Tests for shared visualization grid-evaluation helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from pynamit.primitives.field import Field
from pynamit.primitives.grid import Grid
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.postprocess.grid_evaluation import (
    decode_conductance_dataset_to_grids,
    decode_conductance_entry_to_grids,
    evaluate_scalar_coeffs_to_grid,
    evaluate_tangential_coeffs_to_grid_components,
    get_scalar_grid_evaluation_matrix,
    get_tangential_grid_component_matrices,
)


def _build_test_grid() -> Grid:
    lat, lon = np.meshgrid(
        np.linspace(-80.0, 80.0, 5),
        np.linspace(-180.0, 180.0, 6),
        indexing="ij",
    )
    return Grid(lat=lat, lon=lon)


def test_evaluate_scalar_coeffs_to_grid_matches_field_evaluation() -> None:
    basis = SHBasis(3, 3, mean_free=False)
    grid = _build_test_grid()
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(basis.index_length)

    helper_values = evaluate_scalar_coeffs_to_grid(
        coeffs,
        basis,
        grid,
        grid.lat.shape,
        evaluation_matrix=get_scalar_grid_evaluation_matrix(basis, grid),
    )

    field = Field.from_coefficients(basis, coeffs=coeffs, field_type="scalar")
    field_values, _, _ = field.evaluate(r=None, theta=grid.theta, phi=grid.phi)

    assert np.allclose(helper_values, np.asarray(field_values).reshape(grid.lat.shape))


def test_evaluate_mean_free_scalar_coeffs_to_grid_matches_field_evaluation() -> None:
    basis = SHBasis(3, 3, mean_free=False)
    grid = _build_test_grid()
    rng = np.random.default_rng(10)
    coeffs = rng.standard_normal(basis.scalar_index_length(mean_free=True))

    helper_values = evaluate_scalar_coeffs_to_grid(
        coeffs,
        basis,
        grid,
        grid.lat.shape,
        mean_free=True,
        evaluation_matrix=get_scalar_grid_evaluation_matrix(basis, grid, mean_free=True),
    )

    field = Field.from_coefficients(
        basis,
        coeffs=coeffs,
        field_type="scalar",
        mean_free=True,
    )
    field_values, _, _ = field.evaluate(r=None, theta=grid.theta, phi=grid.phi)

    assert np.allclose(helper_values, np.asarray(field_values).reshape(grid.lat.shape))


def test_evaluate_tangential_coeffs_to_grid_components_matches_field_evaluation() -> None:
    basis = SHBasis(3, 3, mean_free=False)
    grid = _build_test_grid()
    rng = np.random.default_rng(1)
    coeffs = rng.standard_normal((2, basis.index_length))
    theta_matrix, phi_matrix = get_tangential_grid_component_matrices(basis, grid)

    helper_theta, helper_phi = evaluate_tangential_coeffs_to_grid_components(
        coeffs,
        basis,
        grid,
        grid.lat.shape,
        theta_evaluation_matrix=theta_matrix,
        phi_evaluation_matrix=phi_matrix,
    )

    field = Field.from_coefficients(basis, coeffs=coeffs, field_type="tangential")
    _, field_theta, field_phi = field.evaluate(r=None, theta=grid.theta, phi=grid.phi)

    assert np.allclose(helper_theta, np.asarray(field_theta).reshape(grid.lat.shape))
    assert np.allclose(helper_phi, np.asarray(field_phi).reshape(grid.lat.shape))


def test_evaluate_mean_free_tangential_coeffs_to_grid_components_matches_field_evaluation() -> None:
    basis = SHBasis(3, 3, mean_free=False)
    grid = _build_test_grid()
    rng = np.random.default_rng(11)
    coeffs = rng.standard_normal((2, basis.scalar_index_length(mean_free=True)))
    theta_matrix, phi_matrix = get_tangential_grid_component_matrices(
        basis,
        grid,
        mean_free=True,
    )

    helper_theta, helper_phi = evaluate_tangential_coeffs_to_grid_components(
        coeffs,
        basis,
        grid,
        grid.lat.shape,
        mean_free=True,
        theta_evaluation_matrix=theta_matrix,
        phi_evaluation_matrix=phi_matrix,
    )

    field = Field.from_coefficients(
        basis,
        coeffs=coeffs,
        field_type="tangential",
        mean_free=True,
    )
    _, field_theta, field_phi = field.evaluate(r=None, theta=grid.theta, phi=grid.phi)

    assert np.allclose(helper_theta, np.asarray(field_theta).reshape(grid.lat.shape))
    assert np.allclose(helper_phi, np.asarray(field_phi).reshape(grid.lat.shape))


def test_decode_conductance_dataset_to_grids_matches_entry_decoder_for_log_sigma() -> None:
    basis = SHBasis(3, 3, mean_free=False)
    grid = _build_test_grid()
    rng = np.random.default_rng(2)
    coeffs_p = rng.standard_normal(basis.index_length)
    coeffs_h = rng.standard_normal(basis.index_length)
    sigma_floor = 1e-3

    conductance_ds = xr.Dataset(
        data_vars={
            "SH_logSigmaP": (("time", "coeff"), coeffs_p.reshape(1, -1)),
            "SH_logSigmaH": (("time", "coeff"), coeffs_h.reshape(1, -1)),
        },
        coords={"time": [0], "coeff": np.arange(basis.index_length)},
    )

    decoded_from_dataset = decode_conductance_dataset_to_grids(
        conductance_ds,
        t_idx=0,
        storage_basis=basis,
        grid=grid,
        target_shape=grid.lat.shape,
        sigma_floor=sigma_floor,
    )

    decoded_from_entry = decode_conductance_entry_to_grids(
        {"logSigmaP": coeffs_p, "logSigmaH": coeffs_h},
        basis,
        grid,
        grid.lat.shape,
        sigma_floor=sigma_floor,
    )

    for actual, expected in zip(decoded_from_dataset, decoded_from_entry):
        assert np.allclose(actual, expected)
