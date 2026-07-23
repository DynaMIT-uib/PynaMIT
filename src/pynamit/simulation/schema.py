"""Simulation storage schema construction.

This module centralizes the basis and ``FieldSpace`` choices used by
``Simulation`` for persisted input and output time series.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pynamit.fields import FieldSpace
from pynamit.simulation.config import PROJECTION_BASIS_KEYS, SimulationConfig
from pynamit.sphere import (
    BasisView,
    CSBasis,
    SHBasis,
    SolidHarmonics,
    SphericalBasis,
    SurfaceOperators,
)

INPUT_VARIABLES = {
    "jr": ("jr",),
    "Br": ("Br",),
    # The artifact key names the physical input category; its canonical
    # stored variables are the two resistance-tensor coefficients.
    "resistance": ("etaP", "etaH"),
    "u": ("u",),
    "Q_eff": ("Q_eff",),
    "E_source": ("E_source",),
}

INPUT_FIELD_TYPES = {
    "jr": "scalar",
    "Br": "scalar",
    "resistance": "scalar",
    "u": "tangential",
    "Q_eff": "tangential",
    "E_source": "tangential",
}

OUTPUT_VARIABLES = {
    "state": ("m_ind", "m_imp", "Phi", "W"),
    "steady_state": ("m_ind", "m_imp", "Phi", "W"),
}

INPUT_DATASET_KEYS = tuple(INPUT_VARIABLES)
OUTPUT_DATASET_KEYS = tuple(OUTPUT_VARIABLES)
RUN_ARTIFACT_NAMES = frozenset(
    {"settings", "PFAC_matrix", *INPUT_DATASET_KEYS, *OUTPUT_DATASET_KEYS}
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

    cs_basis: CSBasis
    sh_basis: SHBasis
    mean_free_sh_basis: BasisView
    horizontal_basis: SurfaceOperators
    solid_harmonics: SolidHarmonics
    input_variables: Mapping[str, tuple[str, ...]]
    output_variables: Mapping[str, tuple[str, ...]]
    input_field_spaces: Mapping[str, FieldSpace]
    output_field_spaces: Mapping[str, Mapping[str, FieldSpace]]
    input_projection_bases: Mapping[str, SurfaceOperators]

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
    cs_basis = CSBasis(config.Ncs)
    horizontal_basis = cs_basis if horizontal_basis_kind == "CS" else mean_free_sh_basis
    solid_harmonics = SolidHarmonics(mean_free_sh_basis)

    projection_basis_kinds = {
        key: getattr(config, f"{key}_projection_basis") for key in PROJECTION_BASIS_KEYS
    }
    resistance_projection_basis = projection_basis_kinds["resistance"]

    if horizontal_basis_kind == "CS":
        input_bases = {
            "jr": cs_basis,
            # Boundary Br participates in radial continuation and is
            # therefore stored in the poloidal SH space even when its
            # source samples are remapped through the CS grid.
            "Br": mean_free_sh_basis,
            "resistance": cs_basis,
            "u": cs_basis,
            "Q_eff": cs_basis,
            "E_source": cs_basis,
        }
        input_mean_free = {
            "jr": True,
            "Br": True,
            "resistance": False,
            "u": True,
            "Q_eff": True,
            "E_source": True,
        }
        input_projection_bases = {
            "jr": cs_basis,
            "Br": cs_basis,
            "resistance": cs_basis,
            "u": cs_basis,
            "Q_eff": cs_basis,
            "E_source": cs_basis,
        }
    else:
        projection_bases = {"SH": mean_free_sh_basis, "CS": cs_basis}
        input_bases = {
            "jr": mean_free_sh_basis,
            "Br": mean_free_sh_basis,
            "resistance": (sh_basis if resistance_projection_basis == "SH" else cs_basis),
            "u": mean_free_sh_basis,
            "Q_eff": mean_free_sh_basis,
            "E_source": mean_free_sh_basis,
        }
        input_mean_free = None
        input_projection_bases = {
            "jr": projection_bases[projection_basis_kinds["jr"]],
            "Br": projection_bases[projection_basis_kinds["Br"]],
            "resistance": (sh_basis if resistance_projection_basis == "SH" else cs_basis),
            "u": projection_bases[projection_basis_kinds["u"]],
            "Q_eff": projection_bases[projection_basis_kinds["Q_eff"]],
            "E_source": projection_bases[projection_basis_kinds["E_source"]],
        }

    input_field_spaces = field_spaces_from_bases(
        input_bases, INPUT_FIELD_TYPES, mean_free_by_key=input_mean_free
    )
    poloidal_state_space = FieldSpace.from_representation(
        mean_free_sh_basis, field_type="scalar", mean_free=True
    )
    surface_state_space = FieldSpace.from_representation(
        horizontal_basis, field_type="scalar", mean_free=True
    )
    output_field_spaces = {
        key: {
            "m_ind": poloidal_state_space,
            "m_imp": surface_state_space,
            "Phi": surface_state_space,
            "W": surface_state_space,
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
