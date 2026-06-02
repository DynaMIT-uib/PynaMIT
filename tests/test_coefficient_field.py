"""Tests for coefficient-backed field values."""

import numpy as np
import pytest

from pynamit.primitives.coefficient_field import CoefficientField
from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere import CSBasis, SHBasis


def test_coefficient_field_applies_scalar_mean_free_projection():
    """CoefficientField applies scalar mean-free semantics."""
    basis = CSBasis(4)
    field_space = FieldSpace(basis, field_type="scalar", mean_free=True)
    coeffs = np.linspace(0.0, 1.0, basis.index_length) + 2.0

    field = CoefficientField(field_space, coeffs)

    assert field.field_space is field_space
    assert field.basis is basis
    assert field.mean_free
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
    """CoefficientField rejects wrong coefficient lengths."""
    basis = SHBasis(3, 2, mean_free=True)
    field_space = FieldSpace(basis, field_type="scalar")

    with pytest.raises(ValueError, match="CoefficientField.coeffs"):
        CoefficientField(field_space, np.zeros(basis.index_length + 1))
