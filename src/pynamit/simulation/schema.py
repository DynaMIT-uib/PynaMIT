"""Simulation storage schema construction.

This module centralizes the basis and ``FieldSpace`` choices used by
``Simulation`` for persisted input and output time series.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from kompe import (
    GlobalCSBasis,
    ScalarBasis,
    SHBasis,
    SolidHarmonicOperators,
    SurfaceDifferentialBasis,
)

from pynamit.fields import FieldSpace
from pynamit.simulation.config import PROJECTION_BASIS_KEYS, SimulationConfig

INPUT_VARIABLES = {
    "boundary_jr": ("boundary_jr",),
    "boundary_Br": ("boundary_Br",),
    "conductance": ("log_conductance_magnitude", "log_hall_to_pedersen_ratio"),
    "u": ("u",),
    "Q_eff": ("Q_eff",),
    "E_neutral_wind": ("E_neutral_wind",),
}

INPUT_FIELD_TYPES = {
    "boundary_jr": "scalar",
    "boundary_Br": "scalar",
    "conductance": "scalar",
    "u": "tangential",
    "Q_eff": "tangential",
    "E_neutral_wind": "tangential",
}

WIND_FORCING_INPUTS = frozenset({"u", "Q_eff", "E_neutral_wind"})

INPUT_VARIABLE_ATTRS = {
    "boundary_jr": {
        "boundary_jr": {
            "units": "A m-2",
            "long_name": "upper-boundary outward radial current density",
        }
    },
    "boundary_Br": {
        "boundary_Br": {"units": "T", "long_name": "outer-boundary outward radial magnetic field"}
    },
    "conductance": {
        "log_conductance_magnitude": {
            "units": "1",
            "long_name": "log conductance magnitude relative to 1 S",
        },
        "log_hall_to_pedersen_ratio": {
            "units": "1",
            "long_name": "log Hall-to-Pedersen conductance ratio",
        },
    },
    "u": {"u": {"units": "m s-1", "long_name": "neutral wind velocity"}},
    "Q_eff": {
        "Q_eff": {"units": "A m-1", "long_name": "effective neutral-wind sheet-current forcing"}
    },
    "E_neutral_wind": {
        "E_neutral_wind": {"units": "V m-1", "long_name": "equivalent neutral-wind electric field"}
    },
}

OUTPUT_VARIABLES = {
    "dynamic": ("induced_Br", "boundary_jr", "Phi", "W"),
    "equilibrium": ("induced_Br", "boundary_jr", "Phi", "W"),
}

_OUTPUT_FIELD_ATTRS = {
    "induced_Br": {
        "units": "T",
        "long_name": "outward radial magnetic perturbation at the ionosphere",
    },
    "boundary_jr": {
        "units": "A m-2",
        "long_name": "upper-boundary outward radial current density",
    },
    "Phi": {
        "units": "V m-1",
        "long_name": "curl-free electric potential divided by ionospheric radius",
    },
    "W": {
        "units": "V m-1",
        "long_name": "divergence-free electric potential divided by ionospheric radius",
    },
}
OUTPUT_VARIABLE_ATTRS = {
    key: {name: dict(attrs) for name, attrs in _OUTPUT_FIELD_ATTRS.items()}
    for key in OUTPUT_VARIABLES
}

INPUT_DATASET_KEYS = tuple(INPUT_VARIABLES)
OUTPUT_DATASET_KEYS = tuple(OUTPUT_VARIABLES)
SIMULATION_ARTIFACT_NAMES = frozenset(
    {"settings", "gap_Br_response", *INPUT_DATASET_KEYS, *OUTPUT_DATASET_KEYS}
)


__all__ = [
    "INPUT_DATASET_KEYS",
    "INPUT_VARIABLE_ATTRS",
    "OUTPUT_DATASET_KEYS",
    "OUTPUT_VARIABLE_ATTRS",
    "SIMULATION_ARTIFACT_NAMES",
    "WIND_FORCING_INPUTS",
    "SimulationSchema",
    "build_simulation_schema",
    "field_spaces_from_bases",
]


@dataclass
class SimulationSchema:
    """Basis and field-space choices for one simulation configuration.

    The mappings are ordinary dictionaries, making the complete schema
    easy to inspect interactively. The builder creates them once;
    simulation code treats them as configuration, not mutable state.
    """

    cs_basis: GlobalCSBasis
    sh_basis: SHBasis
    mean_free_sh_basis: SurfaceDifferentialBasis
    horizontal_basis: SurfaceDifferentialBasis
    solid_harmonics: SolidHarmonicOperators
    input_variables: dict[str, tuple[str, ...]]
    output_variables: dict[str, tuple[str, ...]]
    input_field_spaces: dict[str, FieldSpace]
    output_field_spaces: dict[str, dict[str, FieldSpace]]
    input_projection_bases: dict[str, SurfaceDifferentialBasis]


def field_spaces_from_bases(
    bases: Mapping[str, ScalarBasis],
    field_types: Mapping[str, str],
    mean_free_by_key: Mapping[str, bool] | None = None,
) -> dict[str, FieldSpace]:
    """Return field-space descriptors for time-series schemas."""
    if set(bases) != set(field_types):
        raise ValueError("Basis and field-type schemas must use the same keys.")
    if mean_free_by_key is not None and set(mean_free_by_key) != set(bases):
        raise ValueError("Mean-free and basis schemas must use the same keys.")

    field_spaces = {}
    for key, basis in bases.items():
        default_mean_free = (
            basis.omits_constant_mode() if isinstance(basis, SurfaceDifferentialBasis) else False
        )
        field_spaces[key] = FieldSpace(
            basis,
            field_type=field_types[key],
            mean_free=(default_mean_free if mean_free_by_key is None else mean_free_by_key[key]),
        )
    return field_spaces


def build_simulation_schema(config: SimulationConfig, *, operator_cache=None) -> SimulationSchema:
    """Build the basis and storage schema for one ``Simulation``."""
    horizontal_basis_kind = config.horizontal_basis_kind

    sh_basis = SHBasis(config.Nmax, config.Mmax, mean_free=False, operator_cache=operator_cache)
    mean_free_sh_basis = sh_basis.with_mean_free(True)
    cs_basis = GlobalCSBasis(config.Ncs)
    horizontal_basis = cs_basis if horizontal_basis_kind == "CS" else mean_free_sh_basis
    solid_harmonics = SolidHarmonicOperators(mean_free_sh_basis)

    projection_basis_kinds = {
        key: getattr(config, f"{key}_projection_basis") for key in PROJECTION_BASIS_KEYS
    }
    conductance_projection_basis = projection_basis_kinds["conductance"]

    projection_bases = {"SH": mean_free_sh_basis, "CS": cs_basis}
    input_projection_bases = {
        key: projection_bases[kind] for key, kind in projection_basis_kinds.items()
    }
    input_bases = {
        "boundary_jr": horizontal_basis,
        # Boundary Br participates in radial continuation, so it is
        # always stored in poloidal SH space.
        "boundary_Br": mean_free_sh_basis,
        # Conductance has a nonzero mean, so SH needs the full basis.
        "conductance": sh_basis if conductance_projection_basis == "SH" else cs_basis,
        "u": horizontal_basis,
        "Q_eff": horizontal_basis,
        "E_neutral_wind": horizontal_basis,
    }
    input_projection_bases["conductance"] = input_bases["conductance"]
    input_mean_free = {key: key != "conductance" for key in input_bases}

    input_field_spaces = field_spaces_from_bases(
        input_bases, INPUT_FIELD_TYPES, mean_free_by_key=input_mean_free
    )
    poloidal_output_space = FieldSpace(mean_free_sh_basis, field_type="scalar", mean_free=True)
    surface_output_space = FieldSpace(horizontal_basis, field_type="scalar", mean_free=True)
    boundary_current_output_space = FieldSpace(
        horizontal_basis,
        field_type="scalar",
        # In CS space the discrete Laplacian's exact range is not
        # identical to the area-mean projector. Preserve the current
        # produced by the private toroidal potential exactly so it can
        # be inverted without changing the derived sheet current.
        mean_free=False,
    )
    output_field_spaces = {
        key: {
            "induced_Br": poloidal_output_space,
            "boundary_jr": boundary_current_output_space,
            "Phi": surface_output_space,
            "W": surface_output_space,
        }
        for key in OUTPUT_VARIABLES
    }

    return SimulationSchema(
        cs_basis=cs_basis,
        sh_basis=sh_basis,
        mean_free_sh_basis=mean_free_sh_basis,
        horizontal_basis=horizontal_basis,
        solid_harmonics=solid_harmonics,
        input_variables=dict(INPUT_VARIABLES),
        output_variables=dict(OUTPUT_VARIABLES),
        input_field_spaces=input_field_spaces,
        output_field_spaces=output_field_spaces,
        input_projection_bases=input_projection_bases,
    )
