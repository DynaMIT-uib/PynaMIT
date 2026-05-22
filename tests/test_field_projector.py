"""Tests for projection into field coefficient spaces."""

import numpy as np

from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_projector import FieldProjector
from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere import Grid, SHBasis


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return Grid(lat=lat_grid.ravel(), lon=lon_grid.ravel())


def test_field_projector_projects_scalar_grid_values():
    """Scalar projection recovers known coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    evaluator = BasisEvaluator(basis, grid)
    expected = np.zeros(basis.index_length)
    expected[1] = 1.0
    expected[3] = -0.25
    values = evaluator.basis_to_grid(expected)

    projector = FieldProjector(FieldSpace(basis, field_type="scalar"))
    actual = projector.project(values, input_grid=grid, projection_basis=basis)

    np.testing.assert_allclose(actual[0], expected, atol=1e-10)


def test_field_projector_projects_tangential_grid_values():
    """Tangential projection recovers Helmholtz coefficients."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    evaluator = BasisEvaluator(basis, grid)
    expected = np.zeros((2, basis.index_length))
    expected[0, 1] = 1.0
    expected[1, 3] = -0.5
    values = evaluator.basis_to_grid(expected, helmholtz=True)

    projector = FieldProjector(FieldSpace(basis, field_type="tangential"))
    actual = projector.project(values, input_grid=grid, projection_basis=basis)

    np.testing.assert_allclose(actual[0], expected.reshape(-1), atol=1e-10)
