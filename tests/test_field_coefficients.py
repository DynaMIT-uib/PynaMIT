"""Tests for field coefficient values."""

import numpy as np
import pytest

from pynamit.fields import FieldCoefficients
from pynamit.fields import FieldSpace
from pynamit.sphere import CSBasis, Grid, SHBasis


def test_field_coefficients_applies_scalar_mean_free_projection():
    """FieldCoefficients applies scalar mean-free semantics."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    coeffs = np.linspace(0.0, 1.0, basis.index_length) + 2.0

    field = FieldCoefficients(field_space, coeffs)

    assert field.field_space is field_space
    assert field.field_space.representation is basis
    assert field.field_space.mean_free
    np.testing.assert_allclose(basis.scalar_mean(field.array), 0.0, atol=1e-12)
    assert field.array.shape == coeffs.shape
    assert field.array.shape == (basis.index_length,)
    np.testing.assert_allclose(field.to_vector(), field.array.reshape(-1))


def test_field_coefficients_preserves_tangential_shape():
    """Tangential coefficient fields keep their two-component layout."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="tangential", mean_free=True)
    coeffs = np.vstack(
        [
            np.linspace(0.0, 1.0, basis.index_length) + 1.0,
            np.linspace(1.0, 2.0, basis.index_length) - 0.5,
        ]
    )

    field = FieldCoefficients(field_space, coeffs)

    assert field.field_space.field_type == "tangential"
    assert field.array.shape == (2, basis.index_length)
    np.testing.assert_allclose(field.to_vector(), field.array.reshape(-1))
    np.testing.assert_allclose(basis.scalar_mean(field.array), np.zeros(2), atol=1e-12)


def test_field_coefficients_canonicalizes_flat_tangential_coefficients():
    """Flat tangential input is stored as component x coefficient."""
    basis = SHBasis(3, 2)
    field_space = FieldSpace(basis, field_type="tangential")
    coeffs = np.arange(2 * basis.index_length)

    field = FieldCoefficients(field_space, coeffs)

    assert field.field_space.coefficient_shape == (2, basis.index_length)
    assert field.array.shape == field.field_space.coefficient_shape
    np.testing.assert_array_equal(field.array, coeffs.reshape(2, basis.index_length))
    np.testing.assert_array_equal(field.to_vector(), coeffs)


def test_field_coefficients_owns_immutable_numpy_values():
    """External mutation cannot invalidate cached operators."""
    basis = SHBasis(3, 2)
    field_space = FieldSpace(basis)
    source = np.arange(basis.index_length, dtype=float)
    field = FieldCoefficients(field_space, source)
    source[:] = -1.0

    np.testing.assert_array_equal(field.array, np.arange(basis.index_length, dtype=float))
    with pytest.raises((TypeError, ValueError)):
        field.array[0] = 10.0


def test_field_coefficients_validates_coefficient_length():
    """FieldCoefficients rejects wrong coefficient lengths."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="scalar")

    with pytest.raises(ValueError, match="FieldCoefficients.array"):
        FieldCoefficients(field_space, np.zeros(basis.index_length + 1))


def test_field_space_accepts_grid_representation():
    """Grid samples define a field space without becoming a basis."""
    grid = Grid(theta=[30.0, 60.0], phi=[0.0, 90.0])
    field_space = FieldSpace.from_representation(grid)
    field = FieldCoefficients(field_space, [1.0, 2.0])

    assert field.field_space.representation is grid
    assert field_space.index_names == ("point",)
    assert field_space.index_length == grid.size
