"""Tests for basis interface enforcement."""

import pytest

from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.basis import Basis, EvaluableBasis
from pynamit.spherical_harmonics.sh_basis import SHBasis


def test_concrete_bases_implement_basis_interface():
    """Concrete basis classes satisfy the shared metadata interface."""
    sh_basis = SHBasis(3, 3)
    cs_basis = CSBasis(4)

    assert isinstance(sh_basis, Basis)
    assert isinstance(sh_basis, EvaluableBasis)
    assert isinstance(cs_basis, Basis)
    assert not isinstance(cs_basis, EvaluableBasis)
    sh_basis.validate_metadata()
    cs_basis.validate_metadata()


def test_incomplete_basis_subclass_is_rejected():
    """Subclasses must declare the required metadata fields."""

    class IncompleteBasis(Basis):
        kind = "incomplete"

    with pytest.raises(TypeError):
        IncompleteBasis()


def test_evaluable_basis_subclass_must_implement_get_G():
    """Evaluable bases must define grid evaluation."""

    class IncompleteEvaluableBasis(EvaluableBasis):
        kind = "incomplete"
        index_names = ["i"]
        index_length = 1
        index_arrays = [[0]]
        minimum_phi_sampling = 1
        caching = False

    with pytest.raises(TypeError):
        IncompleteEvaluableBasis()
