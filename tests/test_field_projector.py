"""Tests for projection into field coefficient spaces."""

import numpy as np
import pytest

from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.primitives.field_projector import FieldProjector
from pynamit.primitives.field_space import FieldSpace
from pynamit.primitives.timeseries import Timeseries
from pynamit.sphere import CSBasis, Grid, SHBasis


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


def test_field_projector_applies_cs_mean_free_projection():
    """CS coefficient storage can keep full length while enforcing zero mean."""
    basis = CSBasis(4)
    grid = Grid(theta=basis.arr_theta, phi=basis.arr_phi)
    values = np.linspace(0.0, 1.0, basis.index_length) + 3.0

    projector = FieldProjector(
        FieldSpace(basis, field_type="scalar", mean_free=True),
        target_grid_basis=basis,
    )
    actual = projector.project(values, input_grid=grid, projection_basis=basis)

    assert actual.shape == (1, basis.index_length)
    np.testing.assert_allclose(basis.scalar_mean(actual[0]), 0.0, atol=1e-12)


def test_timeseries_exposes_storage_spec_and_projects_mean_free_cs_coefficients():
    """Time-series storage honors FieldSpace metadata for direct coefficients."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    timeseries = Timeseries({"state": field_space}, {"state": {"m_ind": "scalar"}})
    values = np.linspace(0.0, 1.0, basis.index_length) + 2.0

    timeseries.add_entry("state", {"m_ind": values}, time=0.0)

    assert timeseries.get_storage_spec("state") is field_space
    assert timeseries.get_data_var_name("state", "m_ind") == "CS_m_ind"
    stored = timeseries.get_entry("state", 0.0)["m_ind"]
    np.testing.assert_allclose(basis.scalar_mean(stored), 0.0, atol=1e-12)


def test_field_expansion_validates_coefficients_against_field_space():
    """FieldExpansion rejects coefficient vectors with the wrong storage length."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="scalar")

    with pytest.raises(ValueError, match="FieldExpansion.coeffs"):
        FieldExpansion(field_space, coeffs=np.zeros(basis.index_length + 1))
