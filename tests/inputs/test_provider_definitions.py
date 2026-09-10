"""Tests for empirical input-provider definitions."""

from dataclasses import replace

from pynamit.external_inputs.coordinates import (
    LIBRARY_GEOGRAPHIC_110KM,
    PYNAMIT_CENTERED_DIPOLE_110KM,
    PYNAMIT_SPHERICAL_GEO_110KM,
    CoordinateConvention,
    ReferenceSurface,
)
from pynamit.external_inputs.provider_definitions import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
)


def test_provider_specs_are_independent_but_share_coordinate_convention():
    """Provider specs share coordinate semantics."""
    assert CONDUCTANCE_PROVIDER_SPEC is not BOUNDARY_JR_PROVIDER_SPEC
    assert BOUNDARY_JR_PROVIDER_SPEC is not NEUTRAL_WIND_PROVIDER_SPEC
    assert (
        CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
        is BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_convention
        is NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention
        is LIBRARY_GEOGRAPHIC_110KM
    )
    assert (
        CONDUCTANCE_PROVIDER_SPEC.output_coordinate_convention
        is BOUNDARY_JR_PROVIDER_SPEC.output_coordinate_convention
        is NEUTRAL_WIND_PROVIDER_SPEC.output_coordinate_convention
        is PYNAMIT_SPHERICAL_GEO_110KM
    )
    assert CONDUCTANCE_PROVIDER_SPEC.fields == ("hall", "pedersen")
    assert CONDUCTANCE_PROVIDER_SPEC.request_coordinate_views == {
        "model": PYNAMIT_CENTERED_DIPOLE_110KM
    }
    assert BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_views == {
        "model": PYNAMIT_CENTERED_DIPOLE_110KM
    }
    assert not NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_views
    assert BOUNDARY_JR_PROVIDER_SPEC.fields == ("jr",)
    assert NEUTRAL_WIND_PROVIDER_SPEC.fields == ("u_theta", "u_phi")


def test_changing_one_provider_convention_does_not_change_the_others():
    """Let one provider adopt another coordinate convention."""
    another_convention = CoordinateConvention(
        coordinate_system="example_provider_coordinates",
        angular_units="degrees",
        latitude_definition="example",
        longitude_definition="east_positive",
        longitude_wrap="[-180,180)",
        reference_surface=ReferenceSurface(kind="sphere", radius_m=6_500_000.0),
    )
    changed = replace(BOUNDARY_JR_PROVIDER_SPEC, request_coordinate_convention=another_convention)
    assert changed.request_coordinate_convention is another_convention
    assert CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention is LIBRARY_GEOGRAPHIC_110KM
    assert NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention is LIBRARY_GEOGRAPHIC_110KM
