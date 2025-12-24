import pytest
import numpy as np
from pynamit.primitives.field import Field
from pynamit.spherical_harmonics.sh_basis import SHBasis
from pynamit.primitives.grid import Grid


def test_field_expansion_as_field():
    """Test that FieldExpansion behaves like a Field."""

    # Setup standard parameters
    Nmax = 2
    Mmax = 1
    basis = SHBasis(Nmax, Mmax)

    coeffs = np.zeros(basis.index_length)
    coeffs[0] = 1000.0

    # Use Factory method
    field_exp = Field.from_coefficients(basis=basis, coeffs=coeffs, field_type="scalar")

    # 1. Test evaluate directly
    theta = np.array([45.0, 90.0])
    phi = np.array([0.0, 0.0])
    r = np.array([6371.2e3, 6371.2e3])

    v1, v2, v3 = field_exp.evaluate(r, theta, phi)

    # Check shapes
    assert v1.size == 2
    assert v2.size == 2
    assert v3.size == 2

    # Check values
    assert not np.isnan(v1).any()
    assert np.all(v2 == 0)
    assert np.all(v3 == 0)

    # 2. Test integration with DiscreteField via discretize
    eval_grid = Grid(theta=theta, phi=phi)

    # Create DiscreteField
    dfield = field_exp.discretize(eval_grid, r=6371.2e3)

    # Verify properties
    assert isinstance(dfield, Field)
    # Check it acts as discrete
    assert dfield.grid is not None
    assert dfield.v1 is not None
    np.testing.assert_allclose(dfield.vec.r, v1)
    np.testing.assert_allclose(dfield.vec.theta, np.zeros_like(v1))
    np.testing.assert_allclose(dfield.vec.phi, np.zeros_like(v1))


def test_field_expansion_tangential():
    """Test tangential FieldExpansion."""
    Nmax = 2
    Mmax = 1
    basis = SHBasis(Nmax, Mmax)
    coeffs = np.zeros((2, basis.index_length))
    coeffs[0, 0] = 100.0

    field_exp = Field.from_coefficients(basis=basis, coeffs=coeffs, field_type="tangential")

    theta = np.array([45.0])
    phi = np.array([0.0])
    r = np.array([6371.2e3])

    v1, v2, v3 = field_exp.evaluate(r, theta, phi)

    assert np.all(v1 == 0)
    assert not np.all(v2 == 0) or not np.all(v3 == 0)


def test_mainfield_continuous_accessors():
    """Test accessing components of continuous Mainfield."""
    from pynamit.primitives.mainfield import Mainfield

    mf = Mainfield(kind="radial", B0=30000e-9)

    # 1. Access components directly
    br_field = mf.vec.r
    btheta_field = mf.vec.theta

    assert isinstance(br_field, Field)
    assert isinstance(btheta_field, Field)
    assert br_field.component_index == 0

    # 2. Evaluate ComponentField
    r = np.array([6371e3])
    theta = np.array([45.0])
    phi = np.array([0.0])

    # Eval parent
    v1, v2, v3 = mf.evaluate(r, theta, phi)

    # Eval component
    c1, c2, c3 = br_field.evaluate(r, theta, phi)

    # ComponentField evaluates to (scalar_val, 0, 0)
    np.testing.assert_allclose(c1, v1)
    np.testing.assert_allclose(c2, 0)
    np.testing.assert_allclose(c3, 0)

    # 3. Discretize ComponentField
    grid = Grid(theta=theta, phi=phi)
    disc_br = br_field.discretize(grid, r[0])

    assert isinstance(disc_br, Field)
    assert disc_br.grid is not None
    np.testing.assert_allclose(disc_br.v1, v1)
    np.testing.assert_allclose(disc_br.v2, 0)
    np.testing.assert_allclose(disc_br.v3, 0)

    # 4. Check vector access on discrete version
    # DiscreteField.vec.r -> array
    assert isinstance(disc_br.vec.r, np.ndarray)
    np.testing.assert_allclose(disc_br.vec.r, v1)
