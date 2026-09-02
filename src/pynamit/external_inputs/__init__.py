"""External empirical inputs and their coordinate conventions."""

from .coordinates import (
    LIBRARY_GEOGRAPHIC_110KM,
    PYNAMIT_CENTERED_DIPOLE_110KM,
    PYNAMIT_SPHERICAL_GEO_110KM,
    CoordinateConvention,
    ExternalInputCoordinates,
    ReferenceSurface,
    SampleGrid,
)
from .fallback_data import FallbackCollection, ProviderSnapshot
from .provider_definitions import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    InputProviderSpec,
)
from .providers import (
    get_boundary_jr_inputs,
    get_conductance_inputs,
    get_input_source,
    get_wind_inputs,
    native_inputs_available,
    require_native_inputs,
    save_fallback_dataset,
    set_input_source,
)

__all__ = [
    "BOUNDARY_JR_PROVIDER_SPEC",
    "CONDUCTANCE_PROVIDER_SPEC",
    "LIBRARY_GEOGRAPHIC_110KM",
    "NEUTRAL_WIND_PROVIDER_SPEC",
    "PROVIDER_SPECS",
    "PYNAMIT_CENTERED_DIPOLE_110KM",
    "PYNAMIT_SPHERICAL_GEO_110KM",
    "ProviderSnapshot",
    "CoordinateConvention",
    "ExternalInputCoordinates",
    "FallbackCollection",
    "InputProviderSpec",
    "ReferenceSurface",
    "SampleGrid",
    "get_boundary_jr_inputs",
    "get_conductance_inputs",
    "get_input_source",
    "get_wind_inputs",
    "native_inputs_available",
    "require_native_inputs",
    "save_fallback_dataset",
    "set_input_source",
]
