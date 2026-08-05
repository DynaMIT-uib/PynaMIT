"""Simulation storage schema construction.

This module centralizes the basis and ``FieldSpace`` choices used by
``Simulation`` for persisted input and output time series.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from kompe import (
    BasisView,
    GlobalCSBasis,
    SHBasis,
    SolidHarmonicOperators,
    SphericalBasis,
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

OUTPUT_VARIABLES = {
    "dynamic": ("induced_Br", "boundary_jr", "Phi", "W"),
    "equilibrium": ("induced_Br", "boundary_jr", "Phi", "W"),
}

INPUT_DATASET_KEYS = tuple(INPUT_VARIABLES)
OUTPUT_DATASET_KEYS = tuple(OUTPUT_VARIABLES)
RUN_ARTIFACT_NAMES = frozenset(
    {"settings", "gap_Br_response", *INPUT_DATASET_KEYS, *OUTPUT_DATASET_KEYS}
)


__all__ = [
    "INPUT_DATASET_KEYS",
    "OUTPUT_DATASET_KEYS",
    "RUN_ARTIFACT_NAMES",
    "SimulationSchema",
    "build_simulation_schema",
    "field_spaces_from_bases",
]


@dataclass(frozen=True)
class SimulationSchema:
    """Field-space schema for one simulation configuration.

    ``FieldSpace`` mappings are canonical persisted coefficient-space
    metadata for inputs and outputs.
    """

    cs_basis: GlobalCSBasis
    sh_basis: SHBasis
    mean_free_sh_basis: BasisView
    horizontal_basis: SurfaceDifferentialBasis
    solid_harmonics: SolidHarmonicOperators
    input_variables: Mapping[str, tuple[str, ...]]
    output_variables: Mapping[str, tuple[str, ...]]
    input_field_spaces: Mapping[str, FieldSpace]
    output_field_spaces: Mapping[str, Mapping[str, FieldSpace]]
    input_projection_bases: Mapping[str, SurfaceDifferentialBasis]

    def __post_init__(self):
        """Own immutable copies of the canonical schema mappings."""
        for name in (
            "input_variables",
            "output_variables",
            "input_field_spaces",
            "input_projection_bases",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "output_field_spaces",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(variable_spaces))
                    for key, variable_spaces in self.output_field_spaces.items()
                }
            ),
        )


def field_spaces_from_bases(
    bases: Mapping[str, SphericalBasis],
    field_types: Mapping[str, str],
    mean_free_by_key: Mapping[str, bool] | None = None,
) -> dict[str, FieldSpace]:
    """Return field-space descriptors for time-series schemas."""
    if set(bases) != set(field_types):
        raise ValueError("Basis and field-type schemas must use the same keys.")

    field_spaces = {}
    for key, basis in bases.items():
        field_spaces[key] = FieldSpace.from_representation(
            basis,
            field_type=field_types[key],
            mean_free=(
                getattr(basis, "mean_free", False)
                if mean_free_by_key is None
                else mean_free_by_key.get(key, getattr(basis, "mean_free", False))
            ),
        )
    return field_spaces


def build_simulation_schema(config: SimulationConfig, *, operator_cache=None) -> SimulationSchema:
    """Build the basis and storage schema for one ``Simulation``."""
    if not isinstance(config, SimulationConfig):
        raise TypeError("build_simulation_schema requires a SimulationConfig.")
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

    if horizontal_basis_kind == "CS":
        input_bases = {
            "boundary_jr": cs_basis,
            # Boundary Br participates in radial continuation and is
            # therefore stored in the poloidal SH space even when its
            # input samples are remapped through the CS grid.
            "boundary_Br": mean_free_sh_basis,
            "conductance": cs_basis,
            "u": cs_basis,
            "Q_eff": cs_basis,
            "E_neutral_wind": cs_basis,
        }
        input_mean_free = {
            "boundary_jr": True,
            "boundary_Br": True,
            "conductance": False,
            "u": True,
            "Q_eff": True,
            "E_neutral_wind": True,
        }
        input_projection_bases = {
            "boundary_jr": cs_basis,
            "boundary_Br": cs_basis,
            "conductance": cs_basis,
            "u": cs_basis,
            "Q_eff": cs_basis,
            "E_neutral_wind": cs_basis,
        }
    else:
        projection_bases = {"SH": mean_free_sh_basis, "CS": cs_basis}
        input_bases = {
            "boundary_jr": mean_free_sh_basis,
            "boundary_Br": mean_free_sh_basis,
            "conductance": (sh_basis if conductance_projection_basis == "SH" else cs_basis),
            "u": mean_free_sh_basis,
            "Q_eff": mean_free_sh_basis,
            "E_neutral_wind": mean_free_sh_basis,
        }
        input_mean_free = None
        input_projection_bases = {
            "boundary_jr": projection_bases[projection_basis_kinds["boundary_jr"]],
            "boundary_Br": projection_bases[projection_basis_kinds["boundary_Br"]],
            "conductance": (sh_basis if conductance_projection_basis == "SH" else cs_basis),
            "u": projection_bases[projection_basis_kinds["u"]],
            "Q_eff": projection_bases[projection_basis_kinds["Q_eff"]],
            "E_neutral_wind": projection_bases[projection_basis_kinds["E_neutral_wind"]],
        }

    input_field_spaces = field_spaces_from_bases(
        input_bases, INPUT_FIELD_TYPES, mean_free_by_key=input_mean_free
    )
    poloidal_output_space = FieldSpace.from_representation(
        mean_free_sh_basis, field_type="scalar", mean_free=True
    )
    surface_output_space = FieldSpace.from_representation(
        horizontal_basis, field_type="scalar", mean_free=True
    )
    boundary_current_output_space = FieldSpace.from_representation(
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
        input_variables=INPUT_VARIABLES,
        output_variables=OUTPUT_VARIABLES,
        input_field_spaces=input_field_spaces,
        output_field_spaces=output_field_spaces,
        input_projection_bases=input_projection_bases,
    )
