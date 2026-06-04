"""Tests for field transforms into coefficient spaces."""

import numpy as np
import pytest

import pynamit
from pynamit.math import JAX_AVAILABLE, set_backend, to_numpy, use_jax
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_transform import FieldTransform
from pynamit.primitives.field_space import FieldSpace
from pynamit.primitives.timeseries import Timeseries
from pynamit.sphere import CSBasis, Grid, SHBasis


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return Grid(lat=lat_grid.ravel(), lon=lon_grid.ravel())


def test_field_transform_projects_scalar_grid_values():
    """Scalar projection recovers known coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="scalar")
    grid = _regular_grid()
    transform = FieldTransform(field_space, grid)
    expected = np.zeros(basis.index_length)
    expected[1] = 1.0
    expected[3] = -0.25
    values = transform.to_grid(expected)

    actual = transform.project(values, input_grid=grid, projection_basis=basis)

    np.testing.assert_allclose(actual[0], expected, atol=1e-10)


def test_basis_evaluator_uses_field_transform_implementation():
    """Historical BasisEvaluator name wraps FieldTransform."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    evaluator = BasisEvaluator(basis, grid)

    assert isinstance(evaluator, FieldTransform)
    assert pynamit.BasisEvaluator is BasisEvaluator

    coeffs = np.zeros(basis.index_length)
    coeffs[1] = 1.0
    np.testing.assert_allclose(
        evaluator.basis_to_grid(coeffs),
        evaluator.to_grid(coeffs),
    )
    np.testing.assert_allclose(
        evaluator.grid_to_basis(evaluator.basis_to_grid(coeffs)),
        coeffs,
        atol=1e-10,
    )

    aliases = {
        "G": "scalar_coeffs_to_grid",
        "G_th": "scalar_coeffs_to_gridded_theta_derivative",
        "G_ph": "scalar_coeffs_to_gridded_phi_derivative",
        "G_grad": "scalar_coeffs_to_gridded_gradient",
        "G_rxgrad": "scalar_coeffs_to_gridded_rhat_cross_gradient",
        "G_helmholtz": "helmholtz_coeffs_to_gridded_vector",
    }
    for historical_name, descriptive_name in aliases.items():
        assert getattr(evaluator, historical_name) is getattr(evaluator, descriptive_name)

    np.testing.assert_allclose(
        evaluator.G_grad,
        np.stack([evaluator.G_th, evaluator.G_ph]),
    )
    np.testing.assert_allclose(
        evaluator.G_rxgrad,
        np.stack([-evaluator.G_ph, evaluator.G_th]),
    )
    np.testing.assert_allclose(
        evaluator.G_helmholtz[:, :, 0, :],
        -evaluator.G_grad,
    )
    np.testing.assert_allclose(
        evaluator.G_helmholtz[:, :, 1, :],
        evaluator.G_rxgrad,
    )


def test_field_transform_projects_tangential_grid_values():
    """Tangential projection recovers Helmholtz coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="tangential")
    grid = _regular_grid()
    transform = FieldTransform(field_space, grid)
    expected = np.zeros((2, basis.index_length))
    expected[0, 1] = 1.0
    expected[1, 3] = -0.5
    values = transform.to_grid(expected)

    actual = transform.project(values, input_grid=grid, projection_basis=basis)

    np.testing.assert_allclose(actual[0], expected.reshape(-1), atol=1e-10)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_field_transform_to_grid_preserves_jax_backend():
    """Coefficient-to-grid synthesis uses LinearMap backend handling."""
    previous_backend = use_jax()
    try:
        set_backend("jax")
        basis = CSBasis(4)
        grid = Grid(theta=basis.arr_theta, phi=basis.arr_phi, area_weights=basis.unit_area)

        scalar_transform = FieldTransform(FieldSpace(basis, field_type="scalar"), grid)
        scalar_coeffs = np.linspace(0.0, 1.0, basis.index_length)
        scalar_values = scalar_transform.to_grid(scalar_coeffs)
        assert "jax" in type(scalar_values).__module__
        np.testing.assert_allclose(
            to_numpy(scalar_values),
            to_numpy(scalar_transform.scalar_coeffs_to_grid) @ scalar_coeffs,
        )

        vector_transform = FieldTransform(FieldSpace(basis, field_type="tangential"), grid)
        vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
        vector_values = vector_transform.to_grid(vector_coeffs)
        assert "jax" in type(vector_values).__module__
        np.testing.assert_allclose(
            to_numpy(vector_values),
            np.tensordot(
                to_numpy(vector_transform.helmholtz_coeffs_to_gridded_vector),
                vector_coeffs,
                2,
            ),
        )
    finally:
        set_backend(previous_backend)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_field_transform_to_grid_preserves_explicit_jax_coefficients():
    """Explicit JAX coefficients reach the LinearMap apply path."""
    import jax.numpy as jnp

    previous_backend = use_jax()
    try:
        set_backend("numpy")
        basis = CSBasis(4)
        grid = Grid(
            theta=np.asarray(basis.arr_theta),
            phi=np.asarray(basis.arr_phi),
            area_weights=np.asarray(basis.unit_area),
        )
        transform = FieldTransform(FieldSpace(basis, field_type="scalar"), grid)
        coeffs = jnp.linspace(0.0, 1.0, basis.index_length)

        values = transform.to_grid(coeffs)

        assert "jax" in type(values).__module__
        np.testing.assert_allclose(
            to_numpy(values),
            transform.scalar_coeffs_to_grid @ to_numpy(coeffs),
        )
    finally:
        set_backend(previous_backend)


def test_field_transform_applies_cs_mean_free_projection():
    """CS projection keeps full length and enforces zero mean."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    grid = Grid(theta=basis.arr_theta, phi=basis.arr_phi, area_weights=basis.unit_area)
    values = np.linspace(0.0, 1.0, basis.index_length) + 3.0

    transform = FieldTransform(field_space, grid, grid_basis=basis)
    actual = transform.project(values, input_grid=grid, projection_basis=basis)

    assert actual.shape == (1, basis.index_length)
    np.testing.assert_allclose(basis.scalar_mean(actual[0]), 0.0, atol=1e-12)


def test_timeseries_exposes_storage_spec_and_projects_mean_free_cs_coefficients():
    """Time-series storage honors FieldSpace metadata."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    timeseries = Timeseries({"state": field_space}, {"state": ("m_ind",)})
    values = np.linspace(0.0, 1.0, basis.index_length) + 2.0

    timeseries.add_entry("state", {"m_ind": values}, time=0.0)

    assert timeseries.get_storage_spec("state") is field_space
    assert timeseries.get_data_var_name("state", "m_ind") == "CS_m_ind"
    stored = timeseries.get_entry("state", 0.0)["m_ind"]
    np.testing.assert_allclose(basis.scalar_mean(stored), 0.0, atol=1e-12)


def test_timeseries_requires_field_space_and_name_only_variables():
    """Time-series schema keeps field types in FieldSpace only."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar")

    with pytest.raises(TypeError, match="field types belong in FieldSpace"):
        Timeseries({"state": field_space}, {"state": {"m_ind": "scalar"}})

    with pytest.raises(ValueError, match="same keys"):
        Timeseries({"state": field_space}, {"other": ("m_ind",)})
