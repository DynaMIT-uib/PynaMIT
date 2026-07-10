"""Tests for height-integrated ionospheric closure equations."""

import numpy as np

from pynamit.simulation.ionospheric_closure import (
    conductance_to_resistance,
    resistance_to_conductance,
)


def test_pedersen_hall_inversion_is_reversible():
    """Finite Pedersen/Hall pairs invert to their original values."""
    hall = np.array([3.0, 4.0])
    pedersen = np.array([4.0, 3.0])

    etaP, etaH = conductance_to_resistance(hall, pedersen)
    sigmaP, sigmaH = resistance_to_conductance(etaP, etaH)

    np.testing.assert_allclose(sigmaP, np.atleast_2d(pedersen))
    np.testing.assert_allclose(sigmaH, np.atleast_2d(hall))


def test_pedersen_hall_inversion_broadcasts_and_marks_zero_pair_invalid():
    """Hall values broadcast and a zero tensor has no finite inverse."""
    etaP, etaH = conductance_to_resistance(0.0, np.array([2.0, 0.0]))

    np.testing.assert_allclose(etaP[0, 0], 0.5)
    np.testing.assert_allclose(etaH[0, 0], 0.0)
    assert np.isnan(etaP[0, 1])
    assert np.isnan(etaH[0, 1])
