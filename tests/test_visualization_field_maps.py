"""Tests for reusable visualization field maps."""

import numpy as np
import scipy.sparse

from pynamit.visualization.field_maps import (
    evaluate_conductance_coefficients,
    evaluate_conductance_values,
    evaluate_electric_field_coefficients,
    evaluate_joule_from_coefficients,
    evaluate_joule_from_fields,
    evaluate_sheet_current_from_maps,
    evaluate_tangential_coefficients,
    evaluate_wind_coefficients,
)


class ScalingTransform:
    """Small transform stub for field-map tests."""

    def synthesize_scalar(self, coeffs):
        """Return deterministic scalar grid values."""
        return 2.0 * np.asarray(coeffs)

    def synthesize_helmholtz(self, coeffs):
        """Return deterministic tangential grid values."""
        return np.asarray(coeffs) + np.array([[1.0, 2.0], [3.0, 4.0]])


class IdentityHelmholtzTransform:
    """Transform stub preserving Helmholtz coefficient values."""

    def synthesize_helmholtz(self, coeffs):
        """Return coefficients as grid values."""
        return np.asarray(coeffs)


def test_conductance_values_include_resistance_and_conductance():
    """Resistance values are returned with conductance."""
    values = evaluate_conductance_values(
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
    )

    np.testing.assert_allclose(values["etaP"], np.array([2.0, 0.0]))
    np.testing.assert_allclose(values["etaH"], np.array([1.0, 0.0]))
    np.testing.assert_allclose(values["SigmaP"][0], 0.4)
    np.testing.assert_allclose(values["SigmaH"][0], 0.2)
    assert np.isnan(values["SigmaP"][1])
    assert np.isnan(values["SigmaH"][1])


def test_conductance_coefficients_use_transform_before_conversion():
    """Coefficient evaluation uses direct conductance conversion."""
    values = evaluate_conductance_coefficients(
        ScalingTransform(),
        np.array([1.0, 2.0]),
        np.array([0.5, 1.0]),
    )

    np.testing.assert_allclose(values["etaP"], np.array([2.0, 4.0]))
    np.testing.assert_allclose(values["etaH"], np.array([1.0, 2.0]))
    np.testing.assert_allclose(values["SigmaP"], np.array([0.4, 0.2]))
    np.testing.assert_allclose(values["SigmaH"], np.array([0.2, 0.1]))


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
        wind["u_mag"],
        np.sqrt(tangential["theta"] ** 2 + tangential["phi"] ** 2),
    )


def test_saved_e_coefficients_are_radius_scaled_to_volt_potentials():
    """Saved E coefficients scale to volt potentials."""
    radius = 6.5
    saved_phi = np.array([1.0, -2.0])
    saved_w = np.array([3.0, 4.0])

    electric_field = evaluate_electric_field_coefficients(
        IdentityHelmholtzTransform(),
        radius * saved_phi,
        radius * saved_w,
        radius,
    )

    np.testing.assert_allclose(electric_field, np.stack([saved_phi, saved_w]))


def test_joule_field_map_uses_sheet_current_dot_electric_field():
    """Joule evaluation handles direct fields and coefficient maps."""
    sheet_current = np.array([[1.0, 2.0], [3.0, 4.0]])
    electric_field = np.array([[5.0, 6.0], [7.0, 8.0]])

    np.testing.assert_allclose(
        evaluate_joule_from_fields(sheet_current, electric_field),
        np.array([26.0, 44.0]),
    )

    expected_current = evaluate_sheet_current_from_maps(
        m_imp=np.array([1.0, 2.0]),
        m_ind=np.array([3.0, 4.0]),
        m_imp_to_sheet=np.eye(4, 2),
        m_ind_to_sheet=2.0 * np.eye(4, 2),
    )
    joule, field, current = evaluate_joule_from_coefficients(
        ScalingTransform(),
        m_imp=np.array([1.0, 2.0]),
        m_ind=np.array([3.0, 4.0]),
        Phi=np.array([1.0, 1.0]),
        W=np.array([2.0, 2.0]),
        radius=2.0,
        m_imp_to_sheet=np.eye(4, 2),
        m_ind_to_sheet=2.0 * np.eye(4, 2),
    )

    np.testing.assert_allclose(
        field,
        evaluate_electric_field_coefficients(
            ScalingTransform(),
            np.array([1.0, 1.0]),
            np.array([2.0, 2.0]),
            2.0,
        ),
    )
    np.testing.assert_allclose(expected_current, np.array([[7.0, 10.0], [0.0, 0.0]]))
    np.testing.assert_allclose(current, expected_current)
    np.testing.assert_allclose(joule, current[0] * field[0] + current[1] * field[1])


def test_sheet_current_map_accepts_sparse_operators():
    """Visualization field maps use the shared LinearMap adapter."""
    current = evaluate_sheet_current_from_maps(
        m_imp=np.array([1.0, 2.0]),
        m_ind=np.array([3.0, 4.0]),
        m_imp_to_sheet=scipy.sparse.csr_matrix(np.eye(4, 2)),
        m_ind_to_sheet=scipy.sparse.csr_matrix(2.0 * np.eye(4, 2)),
    )

    np.testing.assert_allclose(current, np.array([[7.0, 10.0], [0.0, 0.0]]))
