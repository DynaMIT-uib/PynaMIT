"""PynaMIT depends on Kompe without duplicating its public surface."""

import kompe
import pytest

import pynamit
from pynamit.sphere import BasisEvaluator


def test_only_explicit_sphere_compatibility_spelling_is_retained():
    """The narrow evaluator subclass shares Kompe's implementation."""
    assert issubclass(BasisEvaluator, kompe.SphericalTransform)
    assert pynamit.BasisEvaluator is BasisEvaluator


@pytest.mark.parametrize("name", ["GlobalCSBasis", "RegionalCSMesh", "SECSBasis", "SHBasis"])
def test_kompe_types_are_not_reexported_by_pynamit(name):
    """Spherical machinery is imported from its owning package."""
    assert not hasattr(pynamit, name)
