"""Simulation storage schema construction.

This module centralizes the basis and ``FieldSpace`` choices used by
``Dynamics`` for persisted input and output time series.
"""

from dataclasses import dataclass
from typing import Any

from pynamit.primitives.field_space import FieldSpace
from pynamit.sphere import CSBasis, SHBasis, normalize_horizontal_basis_kind


INPUT_VARIABLES = {
    "jr": {"jr": "scalar"},
    "Br": {"Br": "scalar"},
    "conductance": {"etaP": "scalar", "etaH": "scalar"},
    "u": {"u": "tangential"},
}

OUTPUT_VARIABLES = {
    "state": {"m_ind": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
    "steady_state": {"m_ind": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
}


@dataclass(frozen=True)
class SimulationSchema:
    """Field-space schema for one simulation configuration.

    Input/output storage bases are derived from the corresponding
    ``FieldSpace`` mappings to keep coefficient-space metadata canonical.
    """

    cs_basis: Any
    sh_basis: Any
    sh_basis_mean_free: Any
    horizontal_basis: Any
    radial_continuation_basis: Any
    input_vars: dict[str, dict[str, str]]
    output_vars: dict[str, dict[str, str]]
    input_field_spaces: dict[str, FieldSpace]
    output_field_spaces: dict[str, FieldSpace]
    interpolation_bases: dict[str, Any]

    @property
    def input_storage_bases(self) -> dict[str, Any]:
        """Return input storage bases derived from input field spaces."""
        return storage_bases_from_field_spaces(self.input_field_spaces)

    @property
    def output_storage_bases(self) -> dict[str, Any]:
        """Return output storage bases derived from output field spaces."""
        return storage_bases_from_field_spaces(self.output_field_spaces)


def _setting(settings: Any, name: str) -> Any:
    """Return one setting value from an xarray settings dataset or object."""
    attrs = getattr(settings, "attrs", None)
    if attrs is not None and name in attrs:
        return attrs[name]
    return getattr(settings, name)


def _copy_variable_schema(schema: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Return a shallow copy of nested variable schema dictionaries."""
    return {key: dict(variables) for key, variables in schema.items()}


def storage_bases_from_field_spaces(field_spaces: dict[str, FieldSpace]) -> dict[str, Any]:
    """Return storage bases derived from field-space descriptors."""
    return {key: field_space.basis for key, field_space in field_spaces.items()}


def field_spaces_from_bases(
    storage_bases: dict[str, Any],
    variables: dict[str, dict[str, str]],
    mean_free_by_key: dict[str, bool] | None = None,
) -> dict[str, FieldSpace]:
    """Return field-space descriptors for time-series schemas."""
    field_spaces = {}
    for key, basis in storage_bases.items():
        field_types = {variables[key][var] for var in variables[key]}
        if len(field_types) != 1:
            raise ValueError(
                "Mixed scalar and tangential input (unsupported), or invalid input type"
            )
        field_spaces[key] = FieldSpace.from_basis(
            basis,
            field_type=field_types.pop(),
            mean_free=(
                getattr(basis, "mean_free", False)
                if mean_free_by_key is None
                else mean_free_by_key.get(key, getattr(basis, "mean_free", False))
            ),
        )
    return field_spaces


def build_simulation_schema(settings: Any, horizontal_basis_kind: str) -> SimulationSchema:
    """Build the canonical basis and storage schema for one ``Dynamics`` instance."""
    horizontal_basis_kind = normalize_horizontal_basis_kind(horizontal_basis_kind)

    sh_basis = SHBasis(_setting(settings, "Nmax"), _setting(settings, "Mmax"), mean_free=False)
    sh_basis_mean_free = sh_basis.with_mean_free(True)
    cs_basis = CSBasis(_setting(settings, "Ncs"))
    horizontal_basis = cs_basis if horizontal_basis_kind == "CS" else sh_basis_mean_free
    radial_continuation_basis = (
        horizontal_basis
        if horizontal_basis.supports_radial_potential_operators
        else sh_basis_mean_free
    )

    input_vars = _copy_variable_schema(INPUT_VARIABLES)
    output_vars = _copy_variable_schema(OUTPUT_VARIABLES)

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
        interpolation_bases = dict(input_bases)
    else:
        input_bases = {
            "jr": sh_basis_mean_free,
            "Br": sh_basis_mean_free,
            "conductance": sh_basis,
            "u": sh_basis_mean_free,
        }
        input_mean_free = None
        interpolation_bases = {
            "jr": sh_basis_mean_free if bool(_setting(settings, "vector_jr")) else cs_basis,
            "Br": sh_basis_mean_free if bool(_setting(settings, "vector_Br")) else cs_basis,
            "conductance": sh_basis if bool(_setting(settings, "vector_conductance")) else cs_basis,
            "u": sh_basis_mean_free if bool(_setting(settings, "vector_u")) else cs_basis,
        }

    output_bases = {
        "state": horizontal_basis,
        "steady_state": horizontal_basis,
    }

    input_field_spaces = field_spaces_from_bases(
        input_bases, input_vars, mean_free_by_key=input_mean_free
    )
    output_field_spaces = field_spaces_from_bases(
        output_bases,
        output_vars,
        mean_free_by_key={"state": True, "steady_state": True},
    )

    return SimulationSchema(
        cs_basis=cs_basis,
        sh_basis=sh_basis,
        sh_basis_mean_free=sh_basis_mean_free,
        horizontal_basis=horizontal_basis,
        radial_continuation_basis=radial_continuation_basis,
        input_vars=input_vars,
        output_vars=output_vars,
        input_field_spaces=input_field_spaces,
        output_field_spaces=output_field_spaces,
        interpolation_bases=interpolation_bases,
    )
