"""PynaMIT depends on Kompe without duplicating its public surface."""

import pytest

import pynamit


@pytest.mark.parametrize("name", ["GlobalCSBasis", "RegionalCSMesh", "SECSBasis", "SHBasis"])
def test_kompe_types_are_not_reexported_by_pynamit(name):
    """Spherical machinery is imported from its owning package."""
    assert not hasattr(pynamit, name)
