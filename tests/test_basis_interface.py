"""Tests for basis interface enforcement."""

import importlib.util

import pytest
import numpy as np

import pynamit
from pynamit.primitives.basis_evaluator import (
    BasisEvaluator,
    grid_sqrt_area_weights,
    resolve_sqrt_weights,
)
from pynamit.math import JAX_AVAILABLE, set_backend, to_jax, to_numpy, use_jax
from pynamit.math.tensor_operations import weighted_tensor_pinv
from pynamit.sphere import (
    Basis,
    BasisView,
    CSBasis,
    Grid,
    GridBasis,
    RadialLaplaceContinuation,
    SHBasis,
    SurfaceOperators,
    is_grid_basis,
)


def test_public_sphere_package_is_canonical():
    """Basis types are available from the public sphere package."""
    from pynamit.sphere.cubed_sphere.cs_basis import CSBasis as ConcreteCSBasis
    from pynamit.sphere.core import Basis as CoreBasis
    from pynamit.sphere.spherical_harmonics.sh_basis import SHBasis as ConcreteSHBasis

    assert CSBasis is ConcreteCSBasis
    assert SHBasis is ConcreteSHBasis
    assert Basis is CoreBasis
    assert pynamit.CSBasis is CSBasis
    assert pynamit.SHBasis is SHBasis
    assert pynamit.BasisView is BasisView
    assert importlib.util.find_spec("pynamit.basis") is None
    assert importlib.util.find_spec("pynamit.primitives.basis") is None
    assert importlib.util.find_spec("pynamit.cubed_sphere") is None
    assert importlib.util.find_spec("pynamit.spherical_harmonics") is None
    assert importlib.util.find_spec("pynamit.primitives.grid") is None
    assert importlib.util.find_spec("pynamit.utils") is None


def test_concrete_bases_implement_basis_interface():
    """Concrete basis classes satisfy the shared metadata interface."""
    sh_basis = SHBasis(3, 3)
    cs_basis = CSBasis(4)

    assert isinstance(sh_basis, Basis)
    assert isinstance(sh_basis, SurfaceOperators)
    assert isinstance(cs_basis, Basis)
    assert isinstance(cs_basis, GridBasis)
    assert isinstance(cs_basis, SurfaceOperators)
    assert is_grid_basis(cs_basis)
    assert sh_basis.kind == "SH"
    assert cs_basis.kind == "CS"
    assert cs_basis.index_length == cs_basis.arr_theta.size
    sh_basis.validate_metadata()
    cs_basis.validate_metadata()


def test_basis_capability_designators_are_explicit():
    """Potential-basis capabilities are explicit."""
    sh_basis = SHBasis(3, 2)
    cs_basis = CSBasis(4)

    assert sh_basis.supports_surface_potential_operators
    assert sh_basis.supports_radial_potential_operators
    assert isinstance(sh_basis, RadialLaplaceContinuation)
    assert cs_basis.supports_surface_potential_operators
    assert not cs_basis.supports_radial_potential_operators
    assert not isinstance(cs_basis, RadialLaplaceContinuation)
    assert not hasattr(cs_basis, "external_potential_continuation")
    assert not hasattr(cs_basis, "internal_potential_continuation")
    assert not hasattr(cs_basis, "boundary_potential_discontinuity")
    assert not hasattr(cs_basis, "sheet_current_potential")


def test_basis_coefficient_compatibility_uses_coefficient_space():
    """Compatibility depends on coefficient layout."""
    sh_basis = SHBasis(3, 2)

    assert sh_basis.coefficients_are_compatible_with(SHBasis(3, 2))
    assert sh_basis.coefficients_are_compatible_with(SHBasis(3, 2, backend="scipy"))
    assert not sh_basis.coefficients_are_compatible_with(SHBasis(3, 2, Nmin=0))
    assert not sh_basis.coefficients_are_compatible_with(SHBasis(4, 2))
    assert CSBasis(4).coefficients_are_compatible_with(CSBasis(4))
    assert not CSBasis(4).coefficients_are_compatible_with(CSBasis(6))
    assert not sh_basis.coefficients_are_compatible_with(CSBasis(4))


def test_surface_operator_builders_match_component_matrices():
    """Surface operators assemble the expected component matrices."""
    cs_basis = CSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()

    G = cs_basis.get_scalar_evaluation_matrix(grid)
    G_theta = cs_basis.evaluate_on_grid(grid, derivative="theta")
    G_phi = cs_basis.evaluate_on_grid(grid, derivative="phi")
    gradient = cs_basis.get_surface_gradient_matrix(grid)
    rotated = cs_basis.get_rhat_cross_gradient_matrix(grid)
    helmholtz = cs_basis.get_helmholtz_synthesis_matrix(grid)
    laplacian = cs_basis.laplacian()

    np.testing.assert_allclose(G, cs_basis.evaluate_on_grid(grid))
    np.testing.assert_allclose(gradient, np.array([G_theta, G_phi]))
    np.testing.assert_allclose(rotated, np.array([-G_phi, G_theta]))
    np.testing.assert_allclose(helmholtz[:, :, 0, :], -gradient)
    np.testing.assert_allclose(helmholtz[:, :, 1, :], rotated)
    np.testing.assert_allclose(laplacian, cs_basis.laplacian())


def test_radial_laplace_continuation_matches_sh_formulas():
    """Radial continuation uses the SH Laplace-continuation formulas."""
    sh_basis = SHBasis(3, 2)

    np.testing.assert_allclose(
        sh_basis.external_potential_continuation(2.0, 3.0),
        (2.0 / 3.0) ** (1 - sh_basis.n),
    )
    np.testing.assert_allclose(
        sh_basis.internal_potential_continuation(2.0, 3.0),
        (2.0 / 3.0) ** (sh_basis.n + 2),
    )
    np.testing.assert_allclose(
        sh_basis.boundary_potential_discontinuity,
        2 * sh_basis.n + 1,
    )


def test_csbasis_evaluates_with_finite_difference_derivatives():
    """CSBasis exposes native finite-difference derivative matrices."""
    cs_basis = CSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()

    constant = np.ones(cs_basis.index_length)
    cos_theta = np.cos(np.deg2rad(cs_basis.arr_theta))
    expected_dtheta = -np.sin(np.deg2rad(cs_basis.arr_theta))

    G = cs_basis.evaluate_on_grid(grid)
    G_theta = cs_basis.evaluate_on_grid(grid, derivative="theta")

    np.testing.assert_allclose(G @ constant, constant)
    np.testing.assert_allclose(G_theta @ constant, 0.0, atol=1e-12)
    np.testing.assert_allclose(G_theta @ cos_theta, expected_dtheta, atol=1e-2)


def test_csbasis_native_grid_is_cell_centered_with_cell_areas():
    """Native CS coefficients live at cell centers with cell areas."""
    cs_basis = CSBasis(16)
    block, i, j = cs_basis.get_gridpoints(cs_basis.N)
    expected_xi = cs_basis.xi(i[:, :-1, :-1] + 0.5, cs_basis.N).flatten()
    expected_eta = cs_basis.eta(j[:, :-1, :-1] + 0.5, cs_basis.N).flatten()
    step = np.diff(cs_basis.xi(np.array([0, 1]), cs_basis.N))[0]
    midpoint_area = step**2 * np.sqrt(
        np.linalg.det(cs_basis.get_metric_tensor(expected_xi, expected_eta))
    )

    np.testing.assert_allclose(cs_basis.arr_xi, expected_xi)
    np.testing.assert_allclose(cs_basis.arr_eta, expected_eta)
    np.testing.assert_array_equal(cs_basis.arr_block, block[:, :-1, :-1].flatten())
    assert np.all(cs_basis.unit_area > 0.0)
    np.testing.assert_allclose(np.sum(cs_basis.unit_area), 4 * np.pi)
    assert np.all(np.isfinite(cs_basis.arr_theta))
    assert np.all(np.isfinite(cs_basis.arr_phi))
    assert np.all(np.abs(np.sin(np.deg2rad(cs_basis.arr_theta))) > 1e-12)
    assert np.max(np.abs(cs_basis.unit_area - midpoint_area) / cs_basis.unit_area) < 1e-3


def test_csbasis_non_native_scalar_evaluation_uses_interpolation():
    """CS scalar evaluation matches built-in interpolation."""
    cs_basis = CSBasis(8)
    _, theta, phi = cs_basis.cube2spherical(
        cs_basis.xi(np.array([1.2, 2.3, 3.4, 4.5]), cs_basis.N),
        cs_basis.eta(np.array([1.1, 2.2, 3.1, 4.2]), cs_basis.N),
        np.zeros(4),
        deg=True,
    )
    target = Grid(theta=theta, phi=phi)
    coeffs = np.sin(np.deg2rad(cs_basis.arr_theta))

    G = cs_basis.evaluate_on_grid(target)
    expected = cs_basis.interpolate_scalar(
        coeffs,
        cs_basis.arr_theta,
        cs_basis.arr_phi,
        target.theta,
        target.phi,
    )

    np.testing.assert_allclose(G @ coeffs, expected)
    with pytest.raises(NotImplementedError, match="native cubed-sphere grid"):
        cs_basis.evaluate_on_grid(target, derivative="theta")


def test_csbasis_non_native_helmholtz_uses_vector_interpolation():
    """CS non-native Helmholtz evaluation interpolates vectors."""
    cs_basis = CSBasis(8)
    native = Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi)
    _, theta, phi = cs_basis.cube2spherical(
        cs_basis.xi(np.array([1.2, 2.3, 3.4, 4.5]), cs_basis.N),
        cs_basis.eta(np.array([1.1, 2.2, 3.1, 4.2]), cs_basis.N),
        np.zeros(4),
        deg=True,
    )
    target = Grid(theta=theta, phi=phi)

    rng = np.random.default_rng(20260520)
    coeffs = rng.standard_normal((2, cs_basis.index_length))
    native_helmholtz = cs_basis.get_helmholtz_synthesis_matrix(native)
    target_helmholtz = cs_basis.get_helmholtz_synthesis_matrix(target)
    native_vector = np.tensordot(native_helmholtz, coeffs, 2)
    actual = np.tensordot(target_helmholtz, coeffs, 2)

    expected_east, expected_north, _ = cs_basis.interpolate_vector_components(
        native_vector[1],
        -native_vector[0],
        np.zeros_like(native_vector[0]),
        cs_basis.arr_theta,
        cs_basis.arr_phi,
        target.theta,
        target.phi,
    )
    expected = np.stack([-expected_north, expected_east])

    assert target_helmholtz.shape == (2, target.size, 2, cs_basis.index_length)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_csbasis_multi_vector_interpolation_matches_per_field_calls():
    """CS vector interpolation supports multiple fields at once."""
    cs_basis = CSBasis(8)
    _, theta, phi = cs_basis.cube2spherical(
        cs_basis.xi(np.array([1.2, 2.3, 3.4, 4.5]), cs_basis.N),
        cs_basis.eta(np.array([1.1, 2.2, 3.1, 4.2]), cs_basis.N),
        np.zeros(4),
        deg=True,
    )
    fields_east = np.stack(
        [
            np.sin(np.deg2rad(cs_basis.arr_theta)),
            np.cos(np.deg2rad(cs_basis.arr_phi)),
        ],
        axis=-1,
    )
    fields_north = np.stack(
        [
            np.cos(np.deg2rad(cs_basis.arr_theta)),
            np.sin(np.deg2rad(cs_basis.arr_phi)),
        ],
        axis=-1,
    )
    fields_radial = np.zeros_like(fields_east)

    multi = cs_basis.interpolate_vector_components(
        fields_east,
        fields_north,
        fields_radial,
        cs_basis.arr_theta,
        cs_basis.arr_phi,
        theta,
        phi,
    )
    per_field = [
        cs_basis.interpolate_vector_components(
            fields_east[:, i],
            fields_north[:, i],
            fields_radial[:, i],
            cs_basis.arr_theta,
            cs_basis.arr_phi,
            theta,
            phi,
        )
        for i in range(fields_east.shape[-1])
    ]

    for component_index in range(3):
        expected = np.stack(
            [field[component_index] for field in per_field],
            axis=-1,
        )
        np.testing.assert_allclose(multi[component_index], expected)


def test_basis_evaluator_contract_g_matches_explicit_products():
    """BasisEvaluator.contract_G contracts G with vectors/matrices."""
    cs_basis = CSBasis(8)
    evaluator = BasisEvaluator(
        cs_basis,
        Grid(theta=cs_basis.arr_theta, phi=cs_basis.arr_phi),
    )
    vector = np.linspace(1.0, 2.0, cs_basis.index_length)
    matrix = np.diag(vector)

    np.testing.assert_allclose(
        evaluator.contract_G(vector),
        evaluator.G * vector.reshape((1, -1)),
    )
    np.testing.assert_allclose(evaluator.contract_G(matrix), evaluator.G @ matrix)
    with pytest.raises(ValueError, match="vector or matrix"):
        evaluator.contract_G(np.zeros((1, 1, 1)))


def test_area_weight_defaults_use_grid_areas_or_sin_theta():
    """Default area weights use CS areas or sin(theta) grid weights."""
    cs_basis = CSBasis(4)
    cs_grid = Grid(
        theta=cs_basis.arr_theta,
        phi=cs_basis.arr_phi,
        area_weights=cs_basis.unit_area,
    )
    regular_grid = Grid(
        theta=np.array([30.0, 90.0, 150.0]),
        phi=np.array([0.0, 90.0, 180.0]),
    )

    np.testing.assert_allclose(
        grid_sqrt_area_weights(cs_grid),
        np.sqrt(cs_basis.unit_area),
    )
    np.testing.assert_allclose(
        grid_sqrt_area_weights(regular_grid),
        np.sqrt(np.sin(np.deg2rad(regular_grid.theta))),
    )


def test_area_weight_option_and_explicit_weights_override():
    """Global area weighting is used only without explicit weights."""
    cs_basis = CSBasis(4)
    grid = Grid(
        theta=cs_basis.arr_theta,
        phi=cs_basis.arr_phi,
        area_weights=cs_basis.unit_area,
    )
    explicit = np.linspace(1.0, 2.0, grid.size)

    unweighted = BasisEvaluator(cs_basis, grid, area_weighted=False)
    weighted = BasisEvaluator(cs_basis, grid, area_weighted=True)
    overridden = BasisEvaluator(
        cs_basis,
        grid,
        sqrt_weights=explicit,
        area_weighted=True,
    )

    assert unweighted.sqrt_weights is None
    np.testing.assert_allclose(weighted.sqrt_weights, np.sqrt(cs_basis.unit_area))
    np.testing.assert_allclose(overridden.sqrt_weights, explicit)
    np.testing.assert_allclose(
        resolve_sqrt_weights(grid, area_weighted=True, vector=True),
        np.tile(np.sqrt(cs_basis.unit_area), (2, 1)),
    )
    assert resolve_sqrt_weights(
        grid,
        sqrt_weights=explicit,
        area_weighted=True,
        vector=True,
    ) is explicit


def test_weighted_tensor_pinv_matches_explicit_weighted_least_squares():
    """Weighted pseudoinverse solves weighted normal equations."""
    A = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    sqrt_weights = np.array([1.0, 1.5, 2.0, 2.5])
    weight_matrix = np.diag(sqrt_weights**2)

    actual = weighted_tensor_pinv(
        A,
        sqrt_weights=sqrt_weights,
        n_leading_flattened=1,
    )
    expected = np.linalg.solve(A.T @ weight_matrix @ A, A.T @ weight_matrix)

    np.testing.assert_allclose(actual, expected)


def test_csbasis_derivatives_match_first_spherical_harmonics():
    """CS derivatives match first-degree sphere functions."""
    cs_basis = CSBasis(8)
    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()
    theta = np.deg2rad(cs_basis.arr_theta)
    phi = np.deg2rad(cs_basis.arr_phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    fields = [
        (x, np.cos(theta) * np.cos(phi), -np.sin(phi), -2 * x),
        (y, np.cos(theta) * np.sin(phi), np.cos(phi), -2 * y),
        (z, -np.sin(theta), np.zeros_like(theta), -2 * z),
    ]

    G_theta = cs_basis.evaluate_on_grid(grid, derivative="theta")
    G_phi = cs_basis.evaluate_on_grid(grid, derivative="phi")
    laplacian = cs_basis.laplacian()

    for values, expected_theta, expected_phi, expected_laplacian in fields:
        np.testing.assert_allclose(G_theta @ values, expected_theta, atol=1e-2)
        np.testing.assert_allclose(G_phi @ values, expected_phi, atol=1e-2)
        np.testing.assert_allclose(laplacian @ values, expected_laplacian, atol=1.2e-1)


def test_csbasis_derivative_convergence_rates_are_reasonable():
    """CS finite differences show expected RMS convergence rates."""

    def rms_errors(N):
        cs_basis = CSBasis(N)
        grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()
        theta = np.deg2rad(cs_basis.arr_theta)
        phi = np.deg2rad(cs_basis.arr_phi)
        sin_theta = np.sin(theta)
        values_l1 = sin_theta * np.cos(phi)
        values_l2 = sin_theta**2 * np.cos(2 * phi)

        theta_error = cs_basis.evaluate_on_grid(grid, derivative="theta") @ values_l1
        theta_error -= np.cos(theta) * np.cos(phi)
        phi_error = cs_basis.evaluate_on_grid(grid, derivative="phi") @ values_l1
        phi_error -= -np.sin(phi)
        laplacian_l1_error = cs_basis.laplacian() @ values_l1
        laplacian_l1_error -= -2 * values_l1
        laplacian_l2_error = cs_basis.laplacian() @ values_l2
        laplacian_l2_error -= -6 * values_l2

        return np.array(
            [
                np.sqrt(np.mean(theta_error**2)),
                np.sqrt(np.mean(phi_error**2)),
                np.sqrt(np.mean(laplacian_l1_error**2)),
                np.sqrt(np.mean(laplacian_l2_error**2)),
            ]
        )

    resolutions = np.array([8, 12, 16])
    h = np.pi / (2 * resolutions)
    errors = np.array([rms_errors(int(N)) for N in resolutions])
    orders = [
        np.polyfit(np.log(h), np.log(errors[:, error_index]), 1)[0]
        for error_index in range(errors.shape[1])
    ]

    assert orders[0] > 1.9
    assert orders[1] > 1.9
    assert orders[2] > 1.4
    assert orders[3] > 1.8


def test_csbasis_mean_free_projection_is_area_weighted_and_operator_preserving():
    """CS scalar gauges use an area-weighted mean-free projection."""
    cs_basis = CSBasis(8)
    rng = np.random.default_rng(20260520)
    values = rng.standard_normal(cs_basis.index_length) + 3.0
    projected = cs_basis.project_scalar_mean_free(values)

    np.testing.assert_allclose(
        cs_basis.scalar_mean_weights,
        cs_basis.unit_area / np.sum(cs_basis.unit_area),
    )
    assert cs_basis.scalar_mean(projected) == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        cs_basis.laplacian() @ projected,
        cs_basis.laplacian() @ values,
        atol=1e-10,
    )

    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()
    helmholtz = np.stack([values, -2.0 * values + 0.5])
    projected_helmholtz = cs_basis.project_helmholtz_mean_free(helmholtz)

    np.testing.assert_allclose(cs_basis.scalar_mean(projected_helmholtz), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.tensordot(cs_basis.get_helmholtz_synthesis_matrix(grid), projected_helmholtz, 2),
        np.tensordot(cs_basis.get_helmholtz_synthesis_matrix(grid), helmholtz, 2),
        atol=1e-10,
    )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_csbasis_mean_free_projection_preserves_jax_arrays():
    """CS gauge projection preserves backend arrays."""
    previous_backend = use_jax()
    try:
        set_backend("jax")
        cs_basis = CSBasis(4)
        values = to_jax(np.arange(cs_basis.index_length, dtype=float))

        projected = cs_basis.project_scalar_mean_free(values)

        assert "jax" in type(projected).__module__
        np.testing.assert_allclose(to_numpy(cs_basis.scalar_mean(projected)), 0.0, atol=1e-14)
    finally:
        set_backend(previous_backend)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_csbasis_surface_operators_preserve_jax_inputs():
    """CS surface operators accept backend arrays."""
    previous_backend = use_jax()
    try:
        set_backend("jax")
        cs_basis = CSBasis(4)
        grid = type(
            "GridLike",
            (),
            {"theta": to_jax(cs_basis.arr_theta), "phi": to_jax(cs_basis.arr_phi)},
        )()
        values = to_jax(np.arange(cs_basis.index_length, dtype=float))

        G = cs_basis.evaluate_on_grid(grid)
        laplacian_values = cs_basis.get_surface_laplacian_operator().matvec(values)

        assert "jax" in type(G).__module__
        assert "jax" in type(cs_basis.laplacian()).__module__
        assert "jax" in type(laplacian_values).__module__
        np.testing.assert_allclose(
            to_numpy(laplacian_values),
            to_numpy(cs_basis.laplacian()) @ to_numpy(values),
        )
    finally:
        set_backend(previous_backend)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed.")
def test_shbasis_surface_operators_preserve_jax_inputs():
    """SH surface operators accept backend arrays."""
    previous_backend = use_jax()
    try:
        set_backend("jax")
        sh_basis = SHBasis(3, 2)
        grid = type(
            "GridLike",
            (),
            {"theta": to_jax(np.array([30.0, 80.0])), "phi": to_jax(np.array([0.0, 45.0]))},
        )()
        values = to_jax(np.arange(sh_basis.index_length, dtype=float))

        G = sh_basis.evaluate_on_grid(grid)
        grid_values = sh_basis.get_scalar_evaluation_operator(grid).matvec(values)
        shifted = sh_basis.get_external_potential_continuation_operator(2.0, 3.0).matvec(values)

        assert "jax" in type(G).__module__
        assert "jax" in type(sh_basis.laplacian()).__module__
        assert "jax" in type(grid_values).__module__
        assert "jax" in type(shifted).__module__
    finally:
        set_backend(previous_backend)


def test_shbasis_mean_free_option_matches_nmin_one_space():
    """Mean-free SH spaces match the Nmin=1 scalar space."""
    nmin_one = SHBasis(3, 2, Nmin=1)
    mean_free = SHBasis(3, 2, mean_free=True)
    full = SHBasis(3, 2, mean_free=False)
    cached_mean_free = full.with_mean_free(True)
    extended = mean_free.get_extended_basis()

    assert isinstance(cached_mean_free, BasisView)
    assert isinstance(cached_mean_free, SurfaceOperators)
    assert cached_mean_free.supports_radial_potential_operators
    assert cached_mean_free.parent_basis is full
    assert cached_mean_free.root_basis is full
    assert mean_free.scalar_fields_are_mean_free_by_construction()
    assert mean_free.Nmin == nmin_one.Nmin == 1
    assert mean_free.index_length == nmin_one.index_length
    assert cached_mean_free.scalar_fields_are_mean_free_by_construction()
    assert cached_mean_free.get_extended_basis() is full
    assert full.with_mean_free(True) is cached_mean_free
    assert not extended.scalar_fields_are_mean_free_by_construction()
    assert extended.Nmin == 0
    assert extended.index_length > mean_free.index_length


def test_shbasis_mean_free_view_slices_parent_operators():
    """Mean-free SH views slice the full parent coefficient space."""
    full = SHBasis(3, 2, mean_free=False)
    view = full.with_mean_free(True)
    direct_mean_free = SHBasis(3, 2, mean_free=True)
    grid = type("GridLike", (), {"theta": np.array([30.0, 80.0]), "phi": np.array([0.0, 45.0])})()

    assert view.index_length == direct_mean_free.index_length
    np.testing.assert_array_equal(view.index_arrays[0], direct_mean_free.index_arrays[0])
    np.testing.assert_array_equal(view.index_arrays[1], direct_mean_free.index_arrays[1])
    np.testing.assert_allclose(view.evaluate_on_grid(grid), full.evaluate_on_grid(grid)[:, 1:])
    np.testing.assert_allclose(
        view.evaluate_on_grid(grid),
        direct_mean_free.evaluate_on_grid(grid),
    )
    np.testing.assert_allclose(view.laplacian(), direct_mean_free.laplacian())
    np.testing.assert_allclose(
        view.external_potential_continuation(2.0, 3.0),
        direct_mean_free.external_potential_continuation(2.0, 3.0),
    )
    np.testing.assert_allclose(
        view.internal_potential_continuation(2.0, 3.0),
        direct_mean_free.internal_potential_continuation(2.0, 3.0),
    )
    np.testing.assert_allclose(
        view.boundary_potential_discontinuity,
        direct_mean_free.boundary_potential_discontinuity,
    )


def test_basis_view_slices_cs_surface_operators():
    """Generic basis views also slice CS coefficient-space operators."""
    cs_basis = CSBasis(8)
    indices = np.arange(0, cs_basis.index_length, 2)
    view = BasisView(cs_basis, indices, view_name="even")
    grid = type("GridLike", (), {"theta": cs_basis.arr_theta, "phi": cs_basis.arr_phi})()

    assert isinstance(view, SurfaceOperators)
    assert view.kind == "CS"
    assert view.index_length == indices.size
    assert view.supports_surface_potential_operators
    assert not view.supports_radial_potential_operators
    np.testing.assert_allclose(view.index_arrays[0], cs_basis.arr_theta[indices])
    np.testing.assert_allclose(view.index_arrays[1], cs_basis.arr_phi[indices])
    np.testing.assert_allclose(
        view.evaluate_on_grid(grid),
        cs_basis.evaluate_on_grid(grid)[:, indices],
    )
    np.testing.assert_allclose(
        view.get_surface_gradient_matrix(grid),
        cs_basis.get_surface_gradient_matrix(grid)[:, :, indices],
    )
    np.testing.assert_allclose(
        view.laplacian(),
        cs_basis.laplacian()[np.ix_(indices, indices)],
    )
    with pytest.raises(NotImplementedError):
        view.external_potential_continuation(2.0, 3.0)
    with pytest.raises(NotImplementedError):
        view.boundary_potential_discontinuity


def test_shbasis_rejects_inconsistent_mean_free_options():
    """Nmin and mean_free must describe the same scalar space."""
    with pytest.raises(ValueError, match="inconsistent scalar-space options"):
        SHBasis(3, 2, Nmin=0, mean_free=True)


def test_incomplete_basis_subclass_is_rejected():
    """Subclasses must declare the required metadata fields."""

    class IncompleteBasis(Basis):
        kind = "incomplete"

    with pytest.raises(TypeError):
        IncompleteBasis()


def test_surface_operator_subclass_must_implement_evaluate_on_grid():
    """Surface-operator bases must define grid evaluation."""

    class IncompleteSurfaceOperators(SurfaceOperators):
        kind = "incomplete"
        index_names = ["i"]
        index_length = 1
        index_arrays = [[0]]
        minimum_phi_sampling = 1
        caching = False

    with pytest.raises(TypeError):
        IncompleteSurfaceOperators()
