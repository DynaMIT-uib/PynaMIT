"""Tests for reusable visualization field maps."""

import numpy as np
import scipy.sparse

from pynamit.simulation.electrodynamics.ionospheric_closure import (
    conductance_from_log_coordinates,
    conductance_to_log_coordinates,
    conductance_to_resistance,
)
from pynamit.visualization.field_maps import (
    evaluate_conductance_coefficients,
    evaluate_conductance_values,
    evaluate_JS_from_maps,
    evaluate_tangential_coefficients,
    evaluate_wind_coefficients,
)
from pynamit.visualization.output_fields import evaluate_JS_coefficients


class ScalingTransform:
    """Small transform stub for field-map tests."""

    def synthesize_scalar(self, coeffs):
        """Return deterministic scalar grid values."""
        return 2.0 * np.asarray(coeffs)

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


def test_conductance_coefficients_use_transform_before_conversion():
    """Synthesize both canonical fields before physical conversion."""
    log_magnitude_coeffs = np.array([0.1, 0.2])
    log_ratio_coeffs = np.array([-0.3, 0.4])
    values = evaluate_conductance_coefficients(
        ScalingTransform(), log_magnitude_coeffs, log_ratio_coeffs
    )
    expected_log_magnitude = 2.0 * log_magnitude_coeffs
    expected_log_ratio = 2.0 * log_ratio_coeffs
    sigmaP, sigmaH = conductance_from_log_coordinates(expected_log_magnitude, expected_log_ratio)
    etaP, etaH = conductance_to_resistance(sigmaP, sigmaH)

    np.testing.assert_allclose(values["etaP"], etaP)
    np.testing.assert_allclose(values["etaH"], etaH)
    np.testing.assert_allclose(values["SigmaP"], sigmaP)
    np.testing.assert_allclose(values["SigmaH"], sigmaH)


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
    current = evaluate_JS_from_maps(
        boundary_jr=np.array([1.0, 2.0]),
        induced_Br=np.array([3.0, 4.0]),
        boundary_jr_to_JS=scipy.sparse.csr_matrix(np.eye(4, 2)),
        induced_Br_to_JS=scipy.sparse.csr_matrix(2.0 * np.eye(4, 2)),
    )

    np.testing.assert_allclose(current, np.array([[7.0, 10.0], [0.0, 0.0]]))


def test_JS_map_includes_optional_boundary_field():
    """Boundary Br contributes through the same current-map adapter."""
    current = evaluate_JS_from_maps(
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

    class CompatibleBasis:
        def coefficients_are_compatible_with(self, other):
            return other is self

    class Transform:
        basis = CompatibleBasis()

    class SimulationGeometry:
        horizontal_basis = Transform.basis

        @staticmethod
        def boundary_jr_to_gridded_JS(transform):
            return np.eye(4, 2)

        @staticmethod
        def induced_Br_to_gridded_JS(transform):
            return 2.0 * np.eye(4, 2)

        @staticmethod
        def boundary_Br_to_gridded_JS(transform):
            return 3.0 * np.eye(4, 2)

    current = evaluate_JS_coefficients(
        SimulationGeometry(),
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        Transform(),
        boundary_Br=np.array([5.0, 6.0]),
    )

    np.testing.assert_allclose(current, np.array([[22.0, 28.0], [0.0, 0.0]]))
