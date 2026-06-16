"""Simulation storage schema construction.

This module centralizes the basis and ``FieldSpace`` choices used by
``Dynamics`` for persisted input and output time series.
"""

from dataclasses import dataclass
from typing import Any

from pynamit.primitives.field_space import FieldSpace
from pynamit.simulation.config import (
    PROJECTION_BASIS_KEYS,
    normalize_horizontal_basis_kind,
    resolve_projection_basis_settings,
    setting_value,
)
from pynamit.sphere import CSBasis, SHBasis, SolidHarmonics


INPUT_VARIABLES = {
    "jr": ("jr",),
    "Br": ("Br",),
    "conductance": ("etaP", "etaH"),
    "u": ("u",),
    "Q_eff": ("Q_eff",),
}

INPUT_FIELD_TYPES = {
    "jr": "scalar",
    "Br": "scalar",
    "conductance": "scalar",
    "u": "tangential",
    "Q_eff": "tangential",
}

OUTPUT_VARIABLES = {
    "state": ("m_ind", "m_imp", "Phi", "W"),
    "steady_state": ("m_ind", "m_imp", "Phi", "W"),
}

OUTPUT_FIELD_TYPES = {"state": "scalar", "steady_state": "scalar"}


__all__ = ["SimulationSchema", "build_simulation_schema", "field_spaces_from_bases"]


@dataclass(frozen=True)
class SimulationSchema:
    """Field-space schema for one simulation configuration.

    ``FieldSpace`` mappings are canonical persisted coefficient-space
    metadata for inputs and outputs.
    """

    cs_basis: Any
    sh_basis: Any
    sh_basis_mean_free: Any
    horizontal_basis: Any
    solid_harmonics: SolidHarmonics
    input_vars: dict[str, tuple[str, ...]]
    output_vars: dict[str, tuple[str, ...]]
    input_field_spaces: dict[str, FieldSpace]
    output_field_spaces: dict[str, FieldSpace]
    input_projection_bases: dict[str, Any]


def _copy_variable_schema(schema: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Return a shallow copy of variable-name schema tuples."""
    return {key: tuple(variables) for key, variables in schema.items()}


def field_spaces_from_bases(
    bases: dict[str, Any],
    field_types: dict[str, str],
    mean_free_by_key: dict[str, bool] | None = None,
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


def build_simulation_schema(
    settings: Any, horizontal_basis_kind: str | None = None
) -> SimulationSchema:
    """Build the basis and storage schema for one ``Dynamics``."""
    if horizontal_basis_kind is None:
        horizontal_basis_kind = setting_value(settings, "horizontal_basis_kind", "SH")
    horizontal_basis_kind = normalize_horizontal_basis_kind(horizontal_basis_kind)

    sh_basis = SHBasis(
        setting_value(settings, "Nmax"), setting_value(settings, "Mmax"), mean_free=False
    )
    sh_basis_mean_free = sh_basis.with_mean_free(True)
    cs_basis = CSBasis(setting_value(settings, "Ncs"))
    horizontal_basis = cs_basis if horizontal_basis_kind == "CS" else sh_basis_mean_free
    solid_harmonics = SolidHarmonics(sh_basis_mean_free)

    input_vars = _copy_variable_schema(INPUT_VARIABLES)
    output_vars = _copy_variable_schema(OUTPUT_VARIABLES)

    projection_settings = resolve_projection_basis_settings(settings, horizontal_basis_kind)
    projection_basis_kinds = {
        key: projection_settings[f"{key}_projection_basis"] for key in PROJECTION_BASIS_KEYS
    }
    conductance_projection_basis = projection_basis_kinds["conductance"]

    if horizontal_basis_kind == "CS":
        input_bases = {
            "jr": cs_basis,
            "Br": cs_basis,
            "conductance": cs_basis,
            "u": cs_basis,
            "Q_eff": cs_basis,
        }
        input_mean_free = {"jr": True, "Br": True, "conductance": False, "u": True, "Q_eff": True}
        input_projection_bases = dict(input_bases)
    else:
        projection_bases = {"SH": sh_basis_mean_free, "CS": cs_basis}
        input_bases = {
            "jr": sh_basis_mean_free,
            "Br": sh_basis_mean_free,
            "conductance": (sh_basis if conductance_projection_basis == "SH" else cs_basis),
            "u": sh_basis_mean_free,
            "Q_eff": sh_basis_mean_free,
        }
        input_mean_free = None
        input_projection_bases = {
            "jr": projection_bases[projection_basis_kinds["jr"]],
            "Br": projection_bases[projection_basis_kinds["Br"]],
            "conductance": (sh_basis if conductance_projection_basis == "SH" else cs_basis),
            "u": projection_bases[projection_basis_kinds["u"]],
            "Q_eff": projection_bases[projection_basis_kinds["Q_eff"]],
        }

    output_bases = {"state": horizontal_basis, "steady_state": horizontal_basis}

    input_field_spaces = field_spaces_from_bases(
        input_bases, INPUT_FIELD_TYPES, mean_free_by_key=input_mean_free
    )
    output_field_spaces = field_spaces_from_bases(
        output_bases, OUTPUT_FIELD_TYPES, mean_free_by_key={"state": True, "steady_state": True}
    )

    return SimulationSchema(
        cs_basis=cs_basis,
        sh_basis=sh_basis,
        sh_basis_mean_free=sh_basis_mean_free,
        horizontal_basis=horizontal_basis,
        solid_harmonics=solid_harmonics,
        input_vars=input_vars,
        output_vars=output_vars,
        input_field_spaces=input_field_spaces,
        output_field_spaces=output_field_spaces,
        input_projection_bases=input_projection_bases,
    )
