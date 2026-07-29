"""Tests for height-integrated ionospheric closure equations."""

import numpy as np
import pytest

from pynamit.math.linear_map import as_linear_map
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    Q_eff_on_grid_from_wind,
    conductance_from_log_coordinates,
    conductance_to_log_coordinates,
    conductance_to_resistance,
    electric_field_on_grid,
    hall_geometry_tensor,
    joule_heating_from_current,
    pedersen_geometry_tensor,
    resistance_from_log_conductance_coordinates,
    resistance_tensor_on_grid,
    resistance_to_conductance,
    resistance_to_log_conductance_coordinates,
    solve_Q_eff_coefficients,
    wind_motional_E_tensor,
)


def test_pedersen_hall_inversion_is_reversible():
    """Finite Pedersen/Hall pairs invert to their original values."""
    hall = np.array([3.0, 4.0])
    pedersen = np.array([4.0, 3.0])

    etaP, etaH = conductance_to_resistance(pedersen, hall)
    sigmaP, sigmaH = resistance_to_conductance(etaP, etaH)

    np.testing.assert_allclose(sigmaP, pedersen)
    np.testing.assert_allclose(sigmaH, hall)


def test_log_conductance_coordinates_preserve_positive_tensor_and_reciprocal_pair():
    """Magnitude/ratio reconstruct conductance and resistance."""
    pedersen = np.array([1.0, 2.0, 8.0])
    hall = np.array([0.5, 3.0, 16.0])

    log_magnitude, log_ratio = conductance_to_log_coordinates(pedersen, hall)
    reconstructed = conductance_from_log_coordinates(log_magnitude, log_ratio)
    resistance = resistance_from_log_conductance_coordinates(log_magnitude, log_ratio)
    expected_resistance = conductance_to_resistance(pedersen, hall)

    np.testing.assert_allclose(reconstructed[0], pedersen)
    np.testing.assert_allclose(reconstructed[1], hall)
    np.testing.assert_allclose(resistance[0], expected_resistance[0])
    np.testing.assert_allclose(resistance[1], expected_resistance[1])
    np.testing.assert_allclose(log_ratio, np.log(hall / pedersen))


def test_resistance_maps_directly_to_the_same_log_conductance_coordinates():
    """Match direct resistance and reciprocal conductance routes."""
    etaP = np.array([0.4, 0.2, 0.01])
    etaH = np.array([0.3, 0.1, 0.02])
    pedersen, hall = resistance_to_conductance(etaP, etaH)

    direct = resistance_to_log_conductance_coordinates(etaP, etaH)
    via_conductance = conductance_to_log_coordinates(pedersen, hall)

    np.testing.assert_allclose(direct[0], via_conductance[0])
    np.testing.assert_allclose(direct[1], via_conductance[1])
    reconstructed = resistance_from_log_conductance_coordinates(*direct)
    np.testing.assert_allclose(reconstructed[0], etaP)
    np.testing.assert_allclose(reconstructed[1], etaH)


def test_log_coordinate_maps_remain_finite_across_extreme_ratios():
    """Log differences avoid overflow in component ratios."""
    etaP = np.array([1e-300, 1e300])
    etaH = np.array([1e300, 1e-300])

    log_magnitude, log_ratio = resistance_to_log_conductance_coordinates(etaP, etaH)

    assert np.all(np.isfinite(log_magnitude))
    assert np.all(np.isfinite(log_ratio))
    reconstructed = resistance_from_log_conductance_coordinates(log_magnitude, log_ratio)
    np.testing.assert_allclose(reconstructed[0], etaP, rtol=1e-13)
    np.testing.assert_allclose(reconstructed[1], etaH, rtol=1e-13)

    log_magnitude, log_ratio = conductance_to_log_coordinates(etaP, etaH)
    assert np.all(np.isfinite(log_magnitude))
    assert np.all(np.isfinite(log_ratio))
    reconstructed = conductance_from_log_coordinates(log_magnitude, log_ratio)
    np.testing.assert_allclose(reconstructed[0], etaP, rtol=1e-13)
    np.testing.assert_allclose(reconstructed[1], etaH, rtol=1e-13)


@pytest.mark.parametrize(
    ("eta_p", "eta_h", "message"),
    [
        ([0.0, 1.0], [1.0, 1.0], "Pedersen"),
        ([1.0, 1.0], [0.0, 1.0], "Hall"),
        ([1.0, np.nan], [1.0, 1.0], "Pedersen"),
    ],
)
def test_log_conductance_coordinates_require_positive_resistance(eta_p, eta_h, message):
    """Resistance log coordinates require positive finite components."""
    with pytest.raises(ValueError, match=message):
        resistance_to_log_conductance_coordinates(eta_p, eta_h)


def test_log_conductance_reference_only_shifts_magnitude():
    """A unit-reference change adds a magnitude-coordinate shift."""
    pedersen = np.array([1.0, 2.0])
    hall = np.array([3.0, 4.0])

    unit_magnitude, unit_ratio = conductance_to_log_coordinates(pedersen, hall)
    scaled_magnitude, scaled_ratio = conductance_to_log_coordinates(
        pedersen, hall, reference_conductance=2.5
    )

    np.testing.assert_allclose(scaled_magnitude, unit_magnitude - np.log(2.5))
    np.testing.assert_allclose(scaled_ratio, unit_ratio)
    reconstructed = conductance_from_log_coordinates(
        scaled_magnitude, scaled_ratio, reference_conductance=2.5
    )
    np.testing.assert_allclose(reconstructed[0], pedersen)
    np.testing.assert_allclose(reconstructed[1], hall)


@pytest.mark.parametrize("reference", [0.0, -1.0, np.nan, np.inf])
def test_log_conductance_coordinates_require_a_physical_reference(reference):
    """Every conversion rejects a nonphysical logarithm reference."""
    with pytest.raises(ValueError, match="reference_conductance"):
        conductance_to_log_coordinates(1.0, 1.0, reference_conductance=reference)
    with pytest.raises(ValueError, match="reference_conductance"):
        conductance_from_log_coordinates(0.0, 0.0, reference_conductance=reference)
    with pytest.raises(ValueError, match="reference_conductance"):
        resistance_from_log_conductance_coordinates(0.0, 0.0, reference_conductance=reference)
    with pytest.raises(ValueError, match="reference_conductance"):
        resistance_to_log_conductance_coordinates(1.0, 1.0, reference_conductance=reference)


@pytest.mark.parametrize(
    ("pedersen", "hall", "message"),
    [
        ([0.0, 1.0], [1.0, 1.0], "Pedersen"),
        ([1.0, 1.0], [0.0, 1.0], "Hall"),
        ([1.0, np.nan], [1.0, 1.0], "Pedersen"),
    ],
)
def test_log_conductance_coordinates_require_positive_finite_inputs(pedersen, hall, message):
    """Log coordinates reject values outside their physical domain."""
    with pytest.raises(ValueError, match=message):
        conductance_to_log_coordinates(pedersen, hall)


def test_pedersen_hall_inversion_broadcasts_and_marks_zero_pair_invalid():
    """Hall values broadcast and a zero tensor has no finite inverse."""
    etaP, etaH = conductance_to_resistance(np.array([2.0, 0.0]), 0.0)

    np.testing.assert_allclose(etaP[0], 0.5)
    np.testing.assert_allclose(etaH[0], 0.0)
    assert np.isnan(etaP[1])
    assert np.isnan(etaH[1])


def test_geometry_tensors_encode_horizontal_ohms_law():
    """Closure geometry follows the magnetic-field direction."""
    btheta = np.array([0.3, -0.4])
    bphi = np.array([0.4, 0.1])
    br = np.sqrt(1.0 - btheta**2 - bphi**2)
    Br = np.array([3.0, -2.0])

    pedersen = pedersen_geometry_tensor(btheta, bphi, br)
    hall = hall_geometry_tensor(br)
    wind_map = wind_motional_E_tensor(Br)

    np.testing.assert_allclose(pedersen, np.swapaxes(pedersen, 0, 1))
    np.testing.assert_allclose(hall, -np.swapaxes(hall, 0, 1))
    wind = np.array([[2.0, 5.0], [7.0, 11.0]])
    np.testing.assert_allclose(
        np.einsum("ijg,jg->ig", wind_map, wind),
        np.array([[-Br[0] * 7.0, -Br[1] * 11.0], [Br[0] * 2.0, Br[1] * 5.0]]),
    )


def test_grid_electric_field_combines_resistive_and_wind_terms():
    """Direct closure evaluation follows E = R J - u cross B."""
    current = np.array([[1.0, 2.0], [3.0, 4.0]])
    resistance = np.array([[[2.0, 3.0], [1.0, -1.0]], [[-1.0, 1.0], [4.0, 5.0]]])
    wind = np.array([[5.0, 6.0], [7.0, 8.0]])
    wind_to_E = np.array([[[0.0, 0.0], [-2.0, 3.0]], [[2.0, -3.0], [0.0, 0.0]]])

    electric_field = electric_field_on_grid(current, resistance, wind=wind, wind_to_E=wind_to_E)

    expected = np.einsum("ijg,jg->ig", resistance, current)
    expected += np.einsum("ijg,jg->ig", wind_to_E, wind)
    np.testing.assert_allclose(electric_field, expected)


def test_joule_heating_is_the_pedersen_part_of_resistive_work():
    """The antisymmetric Hall closure contributes no dissipation."""
    current = np.array([[1.0, 2.0], [3.0, 4.0]])
    etaP = np.array([2.0, 3.0])
    etaH = np.array([5.0, -7.0])
    pedersen = np.array([[[2.0, 4.0], [1.0, 0.0]], [[1.0, 0.0], [3.0, 5.0]]])
    hall = np.array([[[0.0, 0.0], [1.0, 2.0]], [[-1.0, -2.0], [0.0, 0.0]]])
    resistance = resistance_tensor_on_grid(etaP, etaH, pedersen, hall)

    heating = joule_heating_from_current(current, etaP, pedersen)
    resistive_work = np.einsum("ig,ijg,jg->g", current, resistance, current)

    np.testing.assert_allclose(heating, resistive_work)
    assert np.all(heating >= 0.0)


def test_Q_eff_inverts_wind_electric_field_through_resistance():
    """Q_eff reproduces wind E through the resistive closure."""
    wind = np.array([[2.0, -1.0], [3.0, 4.0]])
    wind_to_E = np.array([[[0.0, 0.0], [2.0, 3.0]], [[-2.0, -3.0], [0.0, 0.0]]])
    resistance = np.array([[[4.0, 5.0], [1.0, -2.0]], [[-1.0, 2.0], [3.0, 4.0]]])

    Q_eff = Q_eff_on_grid_from_wind(wind, wind_to_E, resistance)
    E_wind = np.einsum("abg,bg->ag", wind_to_E, wind)
    reconstructed = np.einsum("abg,bg->ag", resistance, Q_eff)

    np.testing.assert_allclose(reconstructed, E_wind)


def test_Q_eff_coefficient_solve_recovers_exact_response():
    """Recover an exactly representable Q_eff response."""
    matrix = np.array([[2.0, 1.0], [-1.0, 3.0], [4.0, -2.0]])
    expected = np.array([1.5, -0.5])
    operator = as_linear_map(matrix)

    coefficients = solve_Q_eff_coefficients(operator, matrix @ expected)

    np.testing.assert_allclose(coefficients, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"reg_lambda": -1.0}, "reg_lambda"), ({"pinv_rtol": np.nan}, "pinv_rtol")],
)
def test_Q_eff_coefficient_solve_rejects_invalid_controls(kwargs, message):
    """Invalid solver controls must not silently change behavior."""
    operator = as_linear_map(np.eye(2))

    with pytest.raises(ValueError, match=message):
        solve_Q_eff_coefficients(operator, np.ones(2), **kwargs)
