"""Tests for the MAGE forcing preparation helpers."""

import numpy as np

from scripts.simulation.mage_prepare_forcing import wrap_longitude_180_value


def test_wrap_longitude_180_value_accepts_arrays():
    """REMIX longitude grids should wrap elementwise."""
    values = np.array([[0.0, 180.0, 181.0], [-181.0, 540.0, -540.0]])

    wrapped = wrap_longitude_180_value(values)

    np.testing.assert_allclose(wrapped, np.array([[0.0, -180.0, -179.0], [179.0, -180.0, -180.0]]))


def test_wrap_longitude_180_value_preserves_scalar_return_type():
    """Scalar callers still get a Python float."""
    wrapped = wrap_longitude_180_value(181.0)

    assert isinstance(wrapped, float)
    assert wrapped == -179.0
