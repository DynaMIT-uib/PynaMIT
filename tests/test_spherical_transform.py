"""Tests for transforms between spherical representations."""

import numpy as np
import pytest

import pynamit
from pynamit.math import JAX_AVAILABLE, set_backend, to_numpy, use_jax
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.primitives.field_space import FieldSpace
from pynamit.primitives.timeseries import Timeseries
from pynamit.sphere import CSBasis, Grid, SHBasis


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return Grid(lat=lat_grid.ravel(), lon=lon_grid.ravel())


def test_spherical_transform_projects_scalar_grid_values():
    """Scalar projection recovers known coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    expected = np.zeros(basis.index_length)
    expected[1] = 1.0
    expected[3] = -0.25
    values = transform.synthesize_scalar(expected)

    actual = transform.project_scalar(values, input_grid=grid, projection_basis=basis)

    np.testing.assert_allclose(actual[0], expected, atol=1e-10)


def test_basis_evaluator_is_spherical_transform_alias():
    """Historical BasisEvaluator name aliases SphericalTransform."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    evaluator = BasisEvaluator(basis, grid)

    assert BasisEvaluator is SphericalTransform
    assert pynamit.BasisEvaluator is BasisEvaluator
    assert pynamit.SphericalTransform is SphericalTransform
    assert not hasattr(pynamit, "FieldTransform")
    assert not hasattr(pynamit, "Basis")

    coeffs = np.zeros(basis.index_length)
    coeffs[1] = 1.0
    np.testing.assert_allclose(
        evaluator.analyze_scalar(evaluator.synthesize_scalar(coeffs)),
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


def test_spherical_transform_projects_tangential_grid_values():
    """Tangential projection recovers Helmholtz coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    expected = np.zeros((2, basis.index_length))
    expected[0, 1] = 1.0
    expected[1, 3] = -0.5
    values = transform.synthesize_helmholtz(expected)

    actual = transform.project_helmholtz(
        values, input_grid=grid, projection_basis=basis
    )

    np.testing.assert_allclose(actual[0], expected.reshape(-1), atol=1e-10)


def test_spherical_transform_least_squares_use_operator_properties():
    """Least-squares setup should not force dense attributes."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)

    scalar_problem = transform.scalar_least_squares_problem
    helmholtz_problem = transform.helmholtz_least_squares_problem

    assert scalar_problem.A[0] is transform.scalar_coeffs_to_grid_operator
    assert helmholtz_problem.A[0] is transform.helmholtz_coeffs_to_gridded_vector_operator
    assert not hasattr(transform, "_scalar_coeffs_to_grid")
    assert not hasattr(transform, "_helmholtz_coeffs_to_gridded_vector")


def test_native_cs_transform_synthesizes_from_sparse_operator_paths(monkeypatch):
    """Native CS synthesis can apply sparse operators."""
    basis = CSBasis(4)
    grid = Grid(theta=basis.arr_theta, phi=basis.arr_phi, area_weights=basis.unit_area)
    transform = SphericalTransform(basis, grid)
    bundle = basis._get_derivative_bundle()
    theta = bundle["theta"].toarray()
    phi = bundle["phi"].toarray()

    scalar_coeffs = np.linspace(0.0, 1.0, basis.index_length)
    vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
    expected_helmholtz = np.stack(
        [
            -theta @ vector_coeffs[0] - phi @ vector_coeffs[1],
            -phi @ vector_coeffs[0] + theta @ vector_coeffs[1],
        ]
    )

    def fail_evaluate_on_grid(*args, **kwargs):
        raise AssertionError("native CS synthesis should use operator paths")

    monkeypatch.setattr(basis, "evaluate_on_grid", fail_evaluate_on_grid)

    np.testing.assert_allclose(transform.synthesize_scalar(scalar_coeffs), scalar_coeffs)
    np.testing.assert_allclose(
        transform.synthesize_scalar(scalar_coeffs, derivative="theta"),
        theta @ scalar_coeffs,
    )
    np.testing.assert_allclose(
        transform.synthesize_scalar(scalar_coeffs, derivative="phi"),
        phi @ scalar_coeffs,
    )
    np.testing.assert_allclose(
        transform.synthesize_helmholtz(vector_coeffs),
        expected_helmholtz,
    )
    assert not hasattr(transform, "_scalar_coeffs_to_grid")
    assert not hasattr(transform, "_helmholtz_coeffs_to_gridded_vector")


def test_spherical_transform_batches_scalar_grid_interpolation(monkeypatch):
    """Scalar projection batches interpolation rows."""
    basis = SHBasis(3, 2, mean_free=True)
    interpolation_basis = CSBasis(8)
    target_grid = Grid(
        theta=interpolation_basis.arr_theta,
        phi=interpolation_basis.arr_phi,
        area_weights=interpolation_basis.unit_area,
    )
    input_grid = _regular_grid()
    values = np.vstack(
        [
            np.sin(np.deg2rad(input_grid.theta)),
            np.cos(np.deg2rad(input_grid.phi)),
        ]
    )
    transform = SphericalTransform(
        basis,
        target_grid,
        interpolation_basis=interpolation_basis,
    )
    calls = 0
    original = interpolation_basis.interpolate_scalar

    def counted_interpolate_scalar(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        interpolation_basis,
        "interpolate_scalar",
        counted_interpolate_scalar,
    )

    projected = transform.project_scalar(
        values,
        input_grid=input_grid,
        projection_basis=interpolation_basis,
    )

    assert calls == 1
    assert projected.shape == (2, basis.index_length)


def test_spherical_transform_batches_helmholtz_grid_interpolation(monkeypatch):
    """Helmholtz projection batches vector interpolation rows."""
    basis = SHBasis(3, 2, mean_free=True)
    interpolation_basis = CSBasis(8)
    target_grid = Grid(
        theta=interpolation_basis.arr_theta,
        phi=interpolation_basis.arr_phi,
        area_weights=interpolation_basis.unit_area,
    )
    input_grid = _regular_grid()
    theta_values = np.vstack(
        [
            np.sin(np.deg2rad(input_grid.theta)),
            np.cos(np.deg2rad(input_grid.theta)),
        ]
    )
    phi_values = np.vstack(
        [
            np.cos(np.deg2rad(input_grid.phi)),
            np.sin(np.deg2rad(input_grid.phi)),
        ]
    )
    values = np.stack([theta_values, phi_values], axis=1)
    transform = SphericalTransform(
        basis,
        target_grid,
        interpolation_basis=interpolation_basis,
    )
    calls = 0
    original = interpolation_basis.interpolate_vector_components

    def counted_interpolate_vector_components(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        interpolation_basis,
        "interpolate_vector_components",
        counted_interpolate_vector_components,
    )

    projected = transform.project_helmholtz(
        values,
        input_grid=input_grid,
        projection_basis=interpolation_basis,
    )

    assert calls == 1
    assert projected.shape == (2, 2 * basis.index_length)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_spherical_transform_synthesis_preserves_jax_backend():
    """Coefficient-to-grid synthesis uses LinearMap backend handling."""
    previous_backend = use_jax()
    try:
        set_backend("jax")
        basis = CSBasis(4)
        grid = Grid(theta=basis.arr_theta, phi=basis.arr_phi, area_weights=basis.unit_area)

        transform = SphericalTransform(basis, grid)
        scalar_coeffs = np.linspace(0.0, 1.0, basis.index_length)
        scalar_values = transform.synthesize_scalar(scalar_coeffs)
        assert "jax" in type(scalar_values).__module__
        np.testing.assert_allclose(
            to_numpy(scalar_values),
            to_numpy(transform.scalar_coeffs_to_grid) @ scalar_coeffs,
        )

        vector_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])
        vector_values = transform.synthesize_helmholtz(vector_coeffs)
        assert "jax" in type(vector_values).__module__
        np.testing.assert_allclose(
            to_numpy(vector_values),
            np.tensordot(
                to_numpy(transform.helmholtz_coeffs_to_gridded_vector),
                vector_coeffs,
                2,
            ),
        )
    finally:
        set_backend(previous_backend)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_spherical_transform_preserves_explicit_jax_coefficients():
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
        transform = SphericalTransform(basis, grid)
        coeffs = jnp.linspace(0.0, 1.0, basis.index_length)

        values = transform.synthesize_scalar(coeffs)

        assert "jax" in type(values).__module__
        np.testing.assert_allclose(
            to_numpy(values),
            transform.scalar_coeffs_to_grid @ to_numpy(coeffs),
        )
    finally:
        set_backend(previous_backend)


def test_field_space_applies_cs_mean_free_after_spherical_projection():
    """Field-space constraints remain separate from projection."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    grid = Grid(theta=basis.arr_theta, phi=basis.arr_phi, area_weights=basis.unit_area)
    values = np.linspace(0.0, 1.0, basis.index_length) + 3.0

    transform = SphericalTransform(basis, grid, interpolation_basis=basis)
    projected = transform.project_scalar(
        values, input_grid=grid, projection_basis=basis
    )
    actual = field_space.project_mean_free(projected[0])

    assert projected.shape == (1, basis.index_length)
    np.testing.assert_allclose(basis.scalar_mean(actual), 0.0, atol=1e-12)


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
