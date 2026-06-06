"""Simulation storage schema construction.

This module centralizes the basis and ``FieldSpace`` choices used by
``Dynamics`` for persisted input and output time series.
"""

from dataclasses import dataclass
from typing import Any

from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere import CSBasis, SHBasis, SolidHarmonics


INPUT_VARIABLES = {
    "jr": ("jr",),
    "Br": ("Br",),
    "conductance": ("etaP", "etaH"),
    "u": ("u",),
}

INPUT_FIELD_TYPES = {
    "jr": "scalar",
    "Br": "scalar",
    "conductance": "scalar",
    "u": "tangential",
}

OUTPUT_VARIABLES = {
    "state": ("m_ind", "m_imp", "Phi", "W"),
    "steady_state": ("m_ind", "m_imp", "Phi", "W"),
}

OUTPUT_FIELD_TYPES = {
    "state": "scalar",
    "steady_state": "scalar",
}


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


_MISSING = object()


def normalize_horizontal_basis_kind(kind: str) -> str:
    """Normalize a simulation horizontal-basis kind."""
    normalized = str(kind).strip().upper()
    if normalized not in {"SH", "CS"}:
        raise ValueError("horizontal_basis_kind must be one of ['CS', 'SH'].")
    return normalized


def _setting(settings: Any, name: str, default: Any = _MISSING) -> Any:
    """Return one setting from an xarray dataset or object."""
    attrs = getattr(settings, "attrs", None)
    if attrs is not None and name in attrs:
        return attrs[name]
    if hasattr(settings, name):
        return getattr(settings, name)
    if default is not _MISSING:
        return default
    raise AttributeError(name)


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


def build_simulation_schema(settings: Any, horizontal_basis_kind: str) -> SimulationSchema:
    """Build the basis and storage schema for one ``Dynamics``."""
    horizontal_basis_kind = normalize_horizontal_basis_kind(horizontal_basis_kind)

    sh_basis = SHBasis(_setting(settings, "Nmax"), _setting(settings, "Mmax"), mean_free=False)
    sh_basis_mean_free = sh_basis.with_mean_free(True)
    cs_basis = CSBasis(_setting(settings, "Ncs"))
    horizontal_basis = cs_basis if horizontal_basis_kind == "CS" else sh_basis_mean_free
    solid_harmonics = SolidHarmonics(sh_basis_mean_free)

    input_vars = _copy_variable_schema(INPUT_VARIABLES)
    output_vars = _copy_variable_schema(OUTPUT_VARIABLES)

    project_conductance = bool(_setting(settings, "project_conductance", True))

    if horizontal_basis_kind == "CS":
        input_bases = {
            "jr": cs_basis,
            "Br": cs_basis,
            "conductance": cs_basis,
            "u": cs_basis,
        }
        input_mean_free = {
            "jr": True,
            "Br": True,
            "conductance": False,
            "u": True,
        }
        input_projection_bases = dict(input_bases)
    else:
        input_bases = {
            "jr": sh_basis_mean_free,
            "Br": sh_basis_mean_free,
            "conductance": sh_basis if project_conductance else cs_basis,
            "u": sh_basis_mean_free,
        }
        input_mean_free = None
        input_projection_bases = {
            "jr": sh_basis_mean_free if bool(_setting(settings, "vector_jr")) else cs_basis,
            "Br": sh_basis_mean_free if bool(_setting(settings, "vector_Br")) else cs_basis,
            "conductance": (
                sh_basis
                if project_conductance and bool(_setting(settings, "vector_conductance"))
                else cs_basis
            ),
            "u": sh_basis_mean_free if bool(_setting(settings, "vector_u")) else cs_basis,
        }

    output_bases = {
        "state": horizontal_basis,
        "steady_state": horizontal_basis,
    }

    input_field_spaces = field_spaces_from_bases(
        input_bases, INPUT_FIELD_TYPES, mean_free_by_key=input_mean_free
    )
    output_field_spaces = field_spaces_from_bases(
        output_bases,
        OUTPUT_FIELD_TYPES,
        mean_free_by_key={"state": True, "steady_state": True},
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
