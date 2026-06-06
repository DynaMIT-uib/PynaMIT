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


def test_spherical_transform_batches_direct_projection():
    """Direct SH projection handles multiple RHS columns at once."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    transform = SphericalTransform(basis, grid)
    scalar_coeffs = np.zeros((2, basis.index_length))
    scalar_coeffs[0, 1] = 1.0
    scalar_coeffs[1, 3] = -0.25
    scalar_values = np.vstack(
        [transform.synthesize_scalar(row) for row in scalar_coeffs]
    )
    vector_coeffs = np.zeros((2, 2, basis.index_length))
    vector_coeffs[0, 0, 1] = 1.0
    vector_coeffs[1, 1, 3] = -0.5
    vector_values = np.stack(
        [transform.synthesize_helmholtz(row) for row in vector_coeffs]
    )

    scalar_actual = transform.project_scalar(
        scalar_values,
        input_grid=grid,
        projection_basis=basis,
    )
    vector_actual = transform.project_helmholtz(
        vector_values,
        input_grid=grid,
        projection_basis=basis,
    )

    np.testing.assert_allclose(scalar_actual, scalar_coeffs, atol=1e-10)
    np.testing.assert_allclose(
        vector_actual,
        vector_coeffs.reshape(2, -1),
        atol=1e-10,
    )


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


def test_spherical_transform_reuses_scalar_grid_remap(monkeypatch):
    """Scalar projection reuses a cached CS remap operator."""
    CSBasis._shared_remap_matrix_cache.clear()
    basis = SHBasis(3, 2, mean_free=True)
    grid_remap_basis = CSBasis(8)
    source_basis = CSBasis(10)
    target_grid = Grid(
        theta=grid_remap_basis.arr_theta,
        phi=grid_remap_basis.arr_phi,
        area_weights=grid_remap_basis.unit_area,
    )
    input_grid = Grid(theta=source_basis.arr_theta, phi=source_basis.arr_phi)
    values = np.vstack(
        [
            np.sin(np.deg2rad(input_grid.theta)),
            np.cos(np.deg2rad(input_grid.phi)),
        ]
    )
    transform = SphericalTransform(
        basis,
        target_grid,
        grid_remap_basis=grid_remap_basis,
    )
    calls = 0
    original = grid_remap_basis._build_scalar_grid_remap_matrix

    def counted_build_scalar_grid_remap_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        grid_remap_basis,
        "_build_scalar_grid_remap_matrix",
        counted_build_scalar_grid_remap_matrix,
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("supported CS remaps should use cached operators")

    monkeypatch.setattr(grid_remap_basis, "interpolate_scalar", fail_interpolate_scalar)

    projected_1 = transform.project_scalar(
        values,
        input_grid=input_grid,
        projection_basis=grid_remap_basis,
    )
    projected_2 = transform.project_scalar(
        values,
        input_grid=input_grid,
        projection_basis=grid_remap_basis,
    )

    assert calls == 1
    assert projected_1.shape == (2, basis.index_length)
    np.testing.assert_allclose(projected_2, projected_1)


def test_spherical_transform_skips_matching_grid_remap(monkeypatch):
    """Projection skips remapping on matching grids."""
    basis = SHBasis(3, 2, mean_free=True)
    grid_remap_basis = CSBasis(8)
    grid = Grid(
        theta=grid_remap_basis.arr_theta,
        phi=grid_remap_basis.arr_phi,
        area_weights=grid_remap_basis.unit_area,
    )
    values = np.vstack(
        [
            np.sin(np.deg2rad(grid.theta)),
            np.cos(np.deg2rad(grid.phi)),
        ]
    )
    transform = SphericalTransform(
        basis,
        grid,
        grid_remap_basis=grid_remap_basis,
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("matching grids should not interpolate")

    monkeypatch.setattr(
        grid_remap_basis,
        "interpolate_scalar",
        fail_interpolate_scalar,
    )

    projected = transform.project_scalar(
        values,
        input_grid=grid,
        projection_basis=grid_remap_basis,
    )

    assert projected.shape == (2, basis.index_length)


def test_spherical_transform_requires_grid_remap_operator():
    """Grid-to-grid projection requires remap operators."""
    basis = SHBasis(3, 2, mean_free=True)
    target_basis = CSBasis(8)
    source_basis = CSBasis(10)
    target_grid = Grid(
        theta=target_basis.arr_theta,
        phi=target_basis.arr_phi,
        area_weights=target_basis.unit_area,
    )
    input_grid = Grid(theta=source_basis.arr_theta, phi=source_basis.arr_phi)
    values = np.sin(np.deg2rad(input_grid.theta))
    transform = SphericalTransform(
        basis,
        target_grid,
        grid_remap_basis=object(),
    )

    with pytest.raises(TypeError, match="scalar_grid_remap_operator"):
        transform.project_scalar(
            values,
            input_grid=input_grid,
            projection_basis=target_basis,
        )


def test_spherical_transform_reuses_helmholtz_grid_remap(monkeypatch):
    """Helmholtz projection reuses a cached CS remap operator."""
    CSBasis._shared_remap_matrix_cache.clear()
    basis = SHBasis(3, 2, mean_free=True)
    grid_remap_basis = CSBasis(8)
    source_basis = CSBasis(10)
    target_grid = Grid(
        theta=grid_remap_basis.arr_theta,
        phi=grid_remap_basis.arr_phi,
        area_weights=grid_remap_basis.unit_area,
    )
    input_grid = Grid(theta=source_basis.arr_theta, phi=source_basis.arr_phi)
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
        grid_remap_basis=grid_remap_basis,
    )
    calls = 0
    original = grid_remap_basis._build_tangential_grid_remap_matrix

    def counted_build_tangential_grid_remap_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        grid_remap_basis,
        "_build_tangential_grid_remap_matrix",
        counted_build_tangential_grid_remap_matrix,
    )

    def fail_interpolate_vector_components(*args, **kwargs):
        raise AssertionError("supported CS remaps should use cached operators")

    monkeypatch.setattr(
        grid_remap_basis,
        "interpolate_vector_components",
        fail_interpolate_vector_components,
    )

    projected_1 = transform.project_helmholtz(
        values,
        input_grid=input_grid,
        projection_basis=grid_remap_basis,
    )
    projected_2 = transform.project_helmholtz(
        values,
        input_grid=input_grid,
        projection_basis=grid_remap_basis,
    )

    assert calls == 1
    assert projected_1.shape == (2, 2 * basis.index_length)
    np.testing.assert_allclose(projected_2, projected_1)


def test_cs_scalar_remap_operator_matches_interpolation():
    """Cached scalar remap matches the legacy CS interpolation."""
    source_basis = CSBasis(8)
    target_basis = CSBasis(6)
    source_grid = Grid(theta=source_basis.arr_theta, phi=source_basis.arr_phi)
    target_grid = Grid(theta=target_basis.arr_theta, phi=target_basis.arr_phi)
    values = (
        np.sin(np.deg2rad(source_grid.theta))
        + 0.25 * np.cos(np.deg2rad(source_grid.phi))
    )

    operator = target_basis.scalar_grid_remap_operator(source_grid, target_grid)
    actual = operator @ values
    expected = target_basis.interpolate_scalar(
        values,
        source_grid.theta,
        source_grid.phi,
        target_grid.theta,
        target_grid.phi,
    )

    assert operator is target_basis.scalar_grid_remap_operator(
        source_grid,
        target_grid,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_cs_tangential_remap_operator_matches_interpolation():
    """Cached tangential remap matches legacy interpolation."""
    source_basis = CSBasis(8)
    target_basis = CSBasis(6)
    source_grid = Grid(theta=source_basis.arr_theta, phi=source_basis.arr_phi)
    target_grid = Grid(theta=target_basis.arr_theta, phi=target_basis.arr_phi)
    theta_component = np.sin(np.deg2rad(source_grid.theta))
    phi_component = np.cos(np.deg2rad(source_grid.phi))
    values = np.vstack([theta_component, phi_component])

    operator = target_basis.tangential_grid_remap_operator(source_grid, target_grid)
    actual = (operator @ values.reshape(-1)).reshape(2, target_grid.size)
    east, north, _ = target_basis.interpolate_vector_components(
        phi_component,
        -theta_component,
        np.zeros_like(theta_component),
        source_grid.theta,
        source_grid.phi,
        target_grid.theta,
        target_grid.phi,
    )
    expected = np.vstack([-north, east])

    assert operator is target_basis.tangential_grid_remap_operator(
        source_grid,
        target_grid,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_cs_tangential_remap_matrix_cache_is_shared(monkeypatch):
    """Equivalent CS remaps share sparse matrix construction."""
    CSBasis._shared_remap_matrix_cache.clear()
    source_basis = CSBasis(8)
    target_basis = CSBasis(6)
    equivalent_target_basis = CSBasis(6)
    source_grid = Grid(theta=source_basis.arr_theta, phi=source_basis.arr_phi)
    target_grid = Grid(theta=target_basis.arr_theta, phi=target_basis.arr_phi)
    values = np.vstack(
        [
            np.sin(np.deg2rad(source_grid.theta)),
            np.cos(np.deg2rad(source_grid.phi)),
        ]
    )

    first_operator = target_basis.tangential_grid_remap_operator(
        source_grid,
        target_grid,
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("equivalent remap matrix should come from shared cache")

    monkeypatch.setattr(
        equivalent_target_basis,
        "_build_tangential_grid_remap_matrix",
        fail_build,
    )

    second_operator = equivalent_target_basis.tangential_grid_remap_operator(
        source_grid,
        target_grid,
    )

    np.testing.assert_allclose(
        second_operator @ values.reshape(-1),
        first_operator @ values.reshape(-1),
    )


def test_cs_non_native_scalar_operator_uses_remap_without_dense_interpolation(
    monkeypatch,
):
    """CS non-native scalar operators use sparse remaps."""
    basis = CSBasis(8)
    _, theta, phi = basis.cube2spherical(
        basis.xi(np.array([1.2, 2.3, 3.4, 4.5]), basis.N),
        basis.eta(np.array([1.1, 2.2, 3.1, 4.2]), basis.N),
        np.zeros(4),
        deg=True,
    )
    target = Grid(theta=theta, phi=phi)
    coeffs = (
        np.sin(np.deg2rad(basis.arr_theta))
        + 0.25 * np.cos(np.deg2rad(basis.arr_phi))
    )
    expected = basis.interpolate_scalar(
        coeffs,
        basis.arr_theta,
        basis.arr_phi,
        target.theta,
        target.phi,
    )

    def fail_interpolate_scalar(*args, **kwargs):
        raise AssertionError("scalar operator should use the remap LinearMap path")

    monkeypatch.setattr(basis, "interpolate_scalar", fail_interpolate_scalar)

    operator = basis.get_scalar_evaluation_operator(target)

    assert operator.output_shape == (target.size,)
    np.testing.assert_allclose(operator.matvec(coeffs), expected, atol=1e-12)


def test_cs_non_native_vector_operators_use_remap_without_dense_interpolation(
    monkeypatch,
):
    """CS non-native vector operators use sparse remaps."""
    basis = CSBasis(8)
    _, theta, phi = basis.cube2spherical(
        basis.xi(np.array([1.2, 2.3, 3.4, 4.5]), basis.N),
        basis.eta(np.array([1.1, 2.2, 3.1, 4.2]), basis.N),
        np.zeros(4),
        deg=True,
    )
    target = Grid(theta=theta, phi=phi)
    scalar_coeffs = (
        np.sin(np.deg2rad(basis.arr_theta))
        + 0.25 * np.cos(np.deg2rad(basis.arr_phi))
    )
    helmholtz_coeffs = np.vstack([scalar_coeffs, scalar_coeffs[::-1]])

    expected_gradient = np.tensordot(
        basis.get_surface_gradient_matrix(target),
        scalar_coeffs,
        axes=1,
    )
    expected_rxgrad = np.tensordot(
        basis.get_rhat_cross_gradient_matrix(target),
        scalar_coeffs,
        axes=1,
    )
    expected_helmholtz = np.tensordot(
        basis.get_helmholtz_synthesis_matrix(target),
        helmholtz_coeffs,
        axes=2,
    )

    def fail_interpolate_vector_components(*args, **kwargs):
        raise AssertionError("vector operator should use the remap LinearMap path")

    monkeypatch.setattr(
        basis,
        "interpolate_vector_components",
        fail_interpolate_vector_components,
    )

    gradient_operator = basis.get_surface_gradient_operator(target)
    rxgrad_operator = basis.get_rhat_cross_gradient_operator(target)
    helmholtz_operator = basis.get_helmholtz_synthesis_operator(target)

    np.testing.assert_allclose(
        gradient_operator.matvec(scalar_coeffs).reshape(2, target.size),
        expected_gradient,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rxgrad_operator.matvec(scalar_coeffs).reshape(2, target.size),
        expected_rxgrad,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        helmholtz_operator.matvec(helmholtz_coeffs.reshape(-1)).reshape(
            2,
            target.size,
        ),
        expected_helmholtz,
        atol=1e-10,
    )


def test_cs_non_native_scalar_analysis_solves_against_remap_operator():
    """CS scalar analysis is identity only on the native grid."""
    basis = CSBasis(4)
    target_basis = CSBasis(6)
    target = Grid(
        theta=target_basis.arr_theta,
        phi=target_basis.arr_phi,
        area_weights=target_basis.unit_area,
    )
    transform = SphericalTransform(basis, target)
    coeff_rows = np.vstack(
        [
            np.sin(np.deg2rad(basis.arr_theta)),
            np.cos(np.deg2rad(basis.arr_phi)),
        ]
    )
    value_rows = np.stack([transform.synthesize_scalar(row) for row in coeff_rows])

    coeffs = transform.analyze_scalar(value_rows[0])
    projected_rows = transform.project_scalar(
        value_rows,
        input_grid=target,
        projection_basis=basis,
    )

    assert coeffs.shape == (basis.index_length,)
    assert projected_rows.shape == coeff_rows.shape
    np.testing.assert_allclose(transform.synthesize_scalar(coeffs), value_rows[0])
    for projected, expected_values in zip(projected_rows, value_rows):
        np.testing.assert_allclose(
            transform.synthesize_scalar(projected),
            expected_values,
        )


def test_cs_non_native_helmholtz_analysis_solves_against_remap_operator():
    """CS Helmholtz analysis is identity only on the native grid."""
    basis = CSBasis(4)
    target_basis = CSBasis(6)
    target = Grid(
        theta=target_basis.arr_theta,
        phi=target_basis.arr_phi,
        area_weights=target_basis.unit_area,
    )
    transform = SphericalTransform(basis, target)
    base = (
        np.sin(np.deg2rad(basis.arr_theta))
        + 0.25 * np.cos(np.deg2rad(basis.arr_phi))
    )
    coeffs = basis.project_helmholtz_mean_free(np.vstack([base, base[::-1]]))
    values = transform.synthesize_helmholtz(coeffs)

    actual = transform.analyze_helmholtz(values)

    assert actual.shape == (2, basis.index_length)
    np.testing.assert_allclose(
        transform.synthesize_helmholtz(actual),
        values,
        atol=1e-10,
    )


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

    transform = SphericalTransform(basis, grid, grid_remap_basis=basis)
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
