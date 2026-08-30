"""Tests for evaluated magnetic-field geometry."""

import numpy as np
from kompe import SphericalGrid
from kompe.constants import EARTH_RADIUS_M
from kompe.math import get_backend

from pynamit.geomagnetism import MagneticFieldEvaluation, MainField


def test_field_evaluation_uses_selected_array_backend():
    """Provider output crosses to the selected backend only once."""
    grid = SphericalGrid(lat=[-60.0, 20.0, 70.0], lon=[0.0, 90.0, -120.0])
    evaluation = MagneticFieldEvaluation(MainField(kind="radial"), grid, EARTH_RADIUS_M)

    assert evaluation.components.shape == (3, grid.size)
    assert evaluation.horizontal_to_apex.shape == (2, 2, grid.size)
    assert evaluation.radial_to_apex.shape == (1, 1, grid.size)

    array_module = type(evaluation.components).__module__.split(".", maxsplit=1)[0]
    expected_modules = {"jax", "jaxlib"} if get_backend() == "jax" else {"numpy"}
    assert array_module in expected_modules

    np.testing.assert_allclose(
        evaluation.magnitude, np.linalg.norm(np.asarray(evaluation.components), axis=0)
    )
