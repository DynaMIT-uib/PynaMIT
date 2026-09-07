"""Tests for reusable visualization field maps."""

import numpy as np
import pytest
import scipy.sparse
from kompe import GlobalCSBasis, SHBasis, SphericalTransform
from kompe.math import as_linear_map, identity_linear_map

from pynamit.results.field_evaluation import (
    evaluate_conductance_coefficients,
    evaluate_conductance_values,
    evaluate_tangential_coefficients,
    evaluate_wind_coefficients,
)
from pynamit.results.output_fields import evaluate_output_coefficients, evaluate_sheet_current
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_to_log_coordinates,
    conductance_to_resistance,
)


class ScalingTransform:
    """Small transform stub for field-map tests."""

    def synthesize_helmholtz(self, coeffs):
        """Return deterministic tangential grid values."""
        return np.asarray(coeffs) + np.array([[1.0, 2.0], [3.0, 4.0]])


def test_conductance_values_include_resistance_and_conductance():
    """Expose conductance and reciprocal resistance."""
    pedersen = np.array([2.0, 4.0])
    hall = np.array([1.0, 2.0])
    log_magnitude, log_ratio = conductance_to_log_coordinates(pedersen, hall)
    values = evaluate_conductance_values(log_magnitude, log_ratio)
    etaP, etaH = conductance_to_resistance(pedersen, hall)

    np.testing.assert_allclose(values["log_conductance_magnitude"], log_magnitude)
    np.testing.assert_allclose(values["log_hall_to_pedersen_ratio"], log_ratio)
    np.testing.assert_allclose(values["etaP"], etaP)
    np.testing.assert_allclose(values["etaH"], etaH)
    np.testing.assert_allclose(values["SigmaP"], pedersen)
    np.testing.assert_allclose(values["SigmaH"], hall)


@pytest.mark.parametrize("basis_kind", ["SH", "CS"])
def test_conductance_coefficients_use_transform_before_conversion(basis_kind):
    """Physical values follow synthesis on SH and native CS grids."""
    cs_basis = GlobalCSBasis(4)
    basis = SHBasis(2, 2, mean_free=False) if basis_kind == "SH" else cs_basis
    transform = SphericalTransform(basis, cs_basis.native_grid)
    log_magnitude_coeffs = np.linspace(-0.1, 0.2, basis.coefficient_count)
    log_ratio_coeffs = np.linspace(-0.3, 0.4, basis.coefficient_count)
    expected = evaluate_conductance_values(
        transform.synthesize_scalar(log_magnitude_coeffs),
        transform.synthesize_scalar(log_ratio_coeffs),
    )
    values = evaluate_conductance_coefficients(transform, log_magnitude_coeffs, log_ratio_coeffs)

    for name, expected_values in expected.items():
        np.testing.assert_allclose(values[name], expected_values)


def test_resistance_evaluation_retains_values_lost_in_intermediate_conductance():
    """Direct evaluation survives intermediate conductance underflow."""
    values = evaluate_conductance_values(np.array([-700.0]), np.array([400.0]))

    assert values["SigmaP"][0] == 0.0
    np.testing.assert_allclose(values["etaP"], np.exp(300.0))
    np.testing.assert_allclose(values["etaH"], np.exp(700.0))


def test_tangential_and_wind_coefficients_share_component_convention():
    """Wind directions are derived from the generic tangential map."""
    coeffs = np.array([[10.0, 20.0], [30.0, 40.0]])

    tangential = evaluate_tangential_coefficients(ScalingTransform(), coeffs)
    wind = evaluate_wind_coefficients(ScalingTransform(), coeffs)

    np.testing.assert_allclose(tangential["theta"], np.array([11.0, 22.0]))
    np.testing.assert_allclose(tangential["phi"], np.array([33.0, 44.0]))
    np.testing.assert_allclose(wind["u_theta"], tangential["theta"])
    np.testing.assert_allclose(wind["u_phi"], tangential["phi"])
    np.testing.assert_allclose(wind["u_north"], -tangential["theta"])
    np.testing.assert_allclose(wind["u_east"], tangential["phi"])
    np.testing.assert_allclose(
        wind["u_mag"], np.sqrt(tangential["theta"] ** 2 + tangential["phi"] ** 2)
    )


def test_JS_map_accepts_sparse_operators():
    """Visualization field maps use the shared LinearMap adapter."""
    current = evaluate_sheet_current(
        boundary_jr=np.array([1.0, 2.0]),
        induced_Br=np.array([3.0, 4.0]),
        boundary_jr_to_JS=scipy.sparse.csr_matrix(np.eye(4, 2)),
        induced_Br_to_JS=scipy.sparse.csr_matrix(2.0 * np.eye(4, 2)),
    )

    np.testing.assert_allclose(current, np.array([[7.0, 10.0], [0.0, 0.0]]))


def test_JS_evaluation_does_not_mutate_input_coefficients():
    """Summing sources preserves input coefficients."""
    coefficients = np.arange(4.0)
    identity = identity_linear_map((4,))

    current = evaluate_sheet_current(
        coefficients, coefficients, boundary_jr_to_JS=identity, induced_Br_to_JS=identity
    )

    np.testing.assert_array_equal(coefficients, np.arange(4.0))
    np.testing.assert_array_equal(current, (2 * coefficients).reshape(2, 2))


def test_JS_map_includes_optional_boundary_field():
    """Boundary Br contributes through the same current-map adapter."""
    current = evaluate_sheet_current(
        boundary_jr=np.array([1.0, 2.0]),
        induced_Br=np.array([3.0, 4.0]),
        boundary_jr_to_JS=np.eye(4, 2),
        induced_Br_to_JS=2.0 * np.eye(4, 2),
        boundary_Br=np.array([5.0, 6.0]),
        boundary_Br_to_JS=3.0 * np.eye(4, 2),
    )

    np.testing.assert_allclose(current, np.array([[22.0, 28.0], [0.0, 0.0]]))


def test_live_JS_evaluation_includes_boundary_field():
    """Live output evaluation uses every physical JS source."""
    fields = evaluate_output_coefficients(
        {"boundary_jr": np.array([1.0, 2.0]), "induced_Br": np.array([3.0, 4.0])},
        transform=None,
        field_names={"JS_theta", "JS_phi"},
        sheet_current_operators={
            "boundary_jr_to_JS": as_linear_map(np.eye(4, 2), output_shape=(2, 2)),
            "induced_Br_to_JS": as_linear_map(2.0 * np.eye(4, 2), output_shape=(2, 2)),
            "boundary_Br_to_JS": as_linear_map(3.0 * np.eye(4, 2), output_shape=(2, 2)),
        },
        boundary_Br=np.array([5.0, 6.0]),
    )
    current = np.stack((fields["JS_theta"], fields["JS_phi"]))

    np.testing.assert_allclose(current, np.array([[22.0, 28.0], [0.0, 0.0]]))


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_field_map_evaluation_preserves_jax_arrays(backend, data_source):
    """Derived result fields stay on the active array backend."""
    import jax.numpy as jnp

    coefficients = jnp.array([0.1, -0.2])
    values = evaluate_conductance_values(coefficients, coefficients)
    current = evaluate_sheet_current(
        coefficients, coefficients, boundary_jr_to_JS=jnp.eye(4, 2), induced_Br_to_JS=jnp.eye(4, 2)
    )

    assert "jax" in type(values["SigmaP"]).__module__
    assert "jax" in type(values["etaP"]).__module__
    assert "jax" in type(current).__module__
