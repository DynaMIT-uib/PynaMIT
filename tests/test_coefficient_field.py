"""Tests for coefficient-backed field values."""

import numpy as np
import pytest

from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.coefficient_field import CoefficientField
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere import CSBasis, Grid, SHBasis


def _regular_grid():
    lat = np.linspace(-70.0, 70.0, 11)
    lon = np.linspace(0.0, 330.0, 12)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return Grid(lat=lat_grid.ravel(), lon=lon_grid.ravel())


def test_coefficient_field_applies_scalar_mean_free_projection():
    """CoefficientField applies FieldSpace's scalar mean-free semantics."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    coeffs = np.linspace(0.0, 1.0, basis.index_length) + 2.0

    field = CoefficientField(field_space, coeffs)

    assert field.field_space is field_space
    np.testing.assert_allclose(basis.scalar_mean(field.coeffs), 0.0, atol=1e-12)
    assert field.coeffs.shape == coeffs.shape


def test_coefficient_field_preserves_tangential_shape():
    """Tangential coefficient fields keep their two-component layout."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="tangential", mean_free=True)
    coeffs = np.vstack(
        [
            np.linspace(0.0, 1.0, basis.index_length) + 1.0,
            np.linspace(1.0, 2.0, basis.index_length) - 0.5,
        ]
    )

    field = CoefficientField(field_space, coeffs)

    assert field.field_type == "tangential"
    assert field.coeffs.shape == (2, basis.index_length)
    np.testing.assert_allclose(
        basis.scalar_mean(field.coeffs), np.zeros(2), atol=1e-12
    )


def test_coefficient_field_validates_coefficient_length():
    """CoefficientField rejects coefficient vectors with the wrong length."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="scalar")

    with pytest.raises(ValueError, match="CoefficientField.coeffs"):
        CoefficientField(field_space, np.zeros(basis.index_length + 1))


def test_field_expansion_reuses_coefficient_field_storage():
    """FieldExpansion is still evaluable, but storage comes from CoefficientField."""
    basis = SHBasis(3, 2, mean_free=True)
    grid = _regular_grid()
    evaluator = BasisEvaluator(basis, grid)
    coeffs = np.zeros(basis.index_length)
    coeffs[1] = 1.0

    field = FieldExpansion(FieldSpace(basis, field_type="scalar"), coeffs=coeffs)

    assert isinstance(field, CoefficientField)
    np.testing.assert_allclose(field.to_grid(evaluator), evaluator.basis_to_grid(coeffs))
