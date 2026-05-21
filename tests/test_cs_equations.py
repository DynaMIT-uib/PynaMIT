"""Tests for cubed sphere equation helpers."""

import numpy as np
import pytest

from pynamit.sphere import CSBasis
from pynamit.math.cs_equations import CSEquations


def test_sph_to_contravariant_cs_validates_grid_size():
    """Input vectors must match the cubed sphere grid."""
    cs_basis = CSBasis(4)
    equations = CSEquations(cs_basis, RI=1.0)
    values = np.zeros(cs_basis.arr_theta.size)

    u1, u2, u3 = equations.sph_to_contravariant_cs(values, values, values)

    assert u1.shape == values.shape
    assert u2.shape == values.shape
    assert u3.shape == values.shape

    with pytest.raises(ValueError, match="Atheta must contain one value per cubed sphere"):
        equations.sph_to_contravariant_cs(values, values[:-1], values)


def test_derivative_matrix_uses_cubed_sphere_resolution():
    """The cached derivative matrix should use CSBasis.N."""
    cs_basis = CSBasis(4)
    equations = CSEquations(cs_basis, RI=1.0)

    calls = []

    def fake_get_diff(N, coordinate):
        calls.append((N, coordinate))
        return (np.eye(cs_basis.arr_theta.size), np.eye(cs_basis.arr_theta.size))

    cs_basis.get_Diff = fake_get_diff

    Dxi, Deta = equations.D

    assert calls == [(cs_basis.N, "both")]
    assert Dxi.shape == (cs_basis.arr_theta.size, cs_basis.arr_theta.size)
    assert Deta.shape == (cs_basis.arr_theta.size, cs_basis.arr_theta.size)
