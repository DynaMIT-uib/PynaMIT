"""Tests for simulation storage schema construction."""

import numpy as np
import pytest
import xarray as xr

from pynamit.simulation.config import (
    SimulationConfig,
    normalize_horizontal_basis_kind,
    normalize_projection_basis_kind,
    resolve_projection_basis_settings,
    setting_value,
)
from pynamit.simulation.schema import build_simulation_schema, field_spaces_from_bases


def _settings(**attrs):
    defaults = {"Nmax": 3, "Mmax": 2, "Ncs": 4}
    defaults.update(attrs)
    return SimulationConfig(**defaults)


def test_schema_builder_requires_normalized_config():
    """Schema construction starts after configuration normalization."""
    with pytest.raises(TypeError, match="SimulationConfig"):
        build_simulation_schema(xr.Dataset(attrs={"Nmax": 3, "Mmax": 2, "Ncs": 4}))


def test_horizontal_basis_kind_is_simulation_policy():
    """Normalize horizontal basis choices within simulation policy."""
    assert normalize_horizontal_basis_kind(" sh ") == "SH"
    assert normalize_horizontal_basis_kind("cs") == "CS"
    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        normalize_horizontal_basis_kind("grid")


def test_projection_basis_kind_is_input_policy():
    """Normalize projection-basis choices within simulation policy."""
    assert normalize_projection_basis_kind(" sh ", name="boundary_jr_projection_basis") == "SH"
    assert normalize_projection_basis_kind("cs", name="u_projection_basis") == "CS"
    with pytest.raises(ValueError, match="boundary_jr_projection_basis"):
        normalize_projection_basis_kind("grid", name="boundary_jr_projection_basis")


def test_projection_basis_settings_resolve_defaults_and_inheritance():
    """Projection-basis settings share one normalization path."""
    settings = {"u_projection_basis": "CS"}

    resolved = resolve_projection_basis_settings(settings, "SH")

    assert resolved == {
        "boundary_jr_projection_basis": "SH",
        "boundary_Br_projection_basis": "SH",
        "conductance_projection_basis": "SH",
        "u_projection_basis": "CS",
        "E_neutral_wind_projection_basis": "SH",
        "Q_eff_projection_basis": "CS",
    }


def test_projection_basis_settings_reject_sh_projection_in_cs_mode():
    """A CS horizontal basis requires matching CS input settings."""
    with pytest.raises(ValueError, match="boundary_jr_projection_basis"):
        resolve_projection_basis_settings({"boundary_jr_projection_basis": "SH"}, "CS")


def test_setting_value_accepts_attrs_and_data_vars():
    """Settings access unwraps saved scalar values at the boundary."""
    attr_settings = xr.Dataset(attrs={"Nmax": np.array(3)})
    data_var_settings = xr.Dataset({"Nmax": ((), np.array(4))})
    mapping_settings = {"Nmax": np.array(5)}

    assert setting_value(attr_settings, "Nmax") == 3
    assert setting_value(data_var_settings, "Nmax") == 4
    assert setting_value(mapping_settings, "Nmax") == 5
    assert setting_value(attr_settings, "missing", "fallback") == "fallback"


def test_sh_schema_uses_mean_free_sh_inputs_and_outputs():
    """SH mode keeps the established mean-free SH storage choices."""
    schema = build_simulation_schema(_settings())

    assert schema.horizontal_basis is schema.mean_free_sh_basis
    assert schema.solid_harmonics.basis is schema.horizontal_basis
    assert schema.input_field_spaces["boundary_jr"].representation is schema.mean_free_sh_basis
    assert schema.input_field_spaces["boundary_Br"].representation is schema.mean_free_sh_basis
    assert schema.input_field_spaces["u"].representation is schema.mean_free_sh_basis
    assert schema.input_field_spaces["Q_eff"].representation is schema.mean_free_sh_basis
    assert schema.input_field_spaces["conductance"].representation is schema.sh_basis
    assert all(
        space.representation is schema.horizontal_basis
        for space in schema.output_field_spaces["dynamic"].values()
    )

    assert schema.input_field_spaces["boundary_jr"].mean_free
    assert schema.input_field_spaces["boundary_Br"].mean_free
    assert schema.input_field_spaces["u"].mean_free
    assert schema.input_field_spaces["Q_eff"].mean_free
    assert not schema.input_field_spaces["conductance"].mean_free
    assert schema.output_field_spaces["dynamic"]["induced_Br"].mean_free
    assert not schema.output_field_spaces["dynamic"]["boundary_jr"].mean_free
    assert schema.output_field_spaces["dynamic"]["Phi"].mean_free
    assert schema.output_field_spaces["dynamic"]["W"].mean_free


def test_cs_schema_separates_poloidal_and_surface_output_spaces():
    """Keep radial magnetic quantities in mean-free SH space."""
    schema = build_simulation_schema(_settings(horizontal_basis_kind="cs"))

    assert schema.horizontal_basis is schema.cs_basis
    assert schema.solid_harmonics.basis is schema.mean_free_sh_basis
    assert schema.input_field_spaces["boundary_Br"].representation is schema.mean_free_sh_basis
    assert all(
        space.representation is schema.cs_basis
        for key, space in schema.input_field_spaces.items()
        if key != "boundary_Br"
    )
    assert all(basis is schema.cs_basis for basis in schema.input_projection_bases.values())

    output_spaces = schema.output_field_spaces["dynamic"]
    assert output_spaces["induced_Br"].representation is schema.mean_free_sh_basis
    assert all(
        output_spaces[name].representation is schema.cs_basis
        for name in ("boundary_jr", "Phi", "W")
    )
    assert output_spaces["induced_Br"].mean_free
    assert not output_spaces["boundary_jr"].mean_free
    assert output_spaces["Phi"].mean_free
    assert output_spaces["W"].mean_free


def test_schema_respects_input_projection_basis_for_sh_mode():
    """Input projection choices are explicit in SH mode."""
    schema = build_simulation_schema(
        _settings(
            boundary_jr_projection_basis="CS",
            boundary_Br_projection_basis="CS",
            conductance_projection_basis="CS",
            u_projection_basis="CS",
            Q_eff_projection_basis="CS",
            E_neutral_wind_projection_basis="CS",
        )
    )

    assert schema.input_field_spaces["boundary_jr"].representation is schema.mean_free_sh_basis
    assert schema.input_field_spaces["conductance"].representation is schema.cs_basis
    assert all(basis is schema.cs_basis for basis in schema.input_projection_bases.values())


def test_sh_schema_can_store_conductance_on_cs_grid():
    """CS conductance storage keeps the SH horizontal basis."""
    schema = build_simulation_schema(_settings(conductance_projection_basis="CS"))

    assert schema.horizontal_basis is schema.mean_free_sh_basis
    assert schema.input_field_spaces["conductance"].representation is schema.cs_basis
    assert schema.input_projection_bases["conductance"] is schema.cs_basis
    assert not schema.input_field_spaces["conductance"].mean_free


def test_field_spaces_from_bases_rejects_invalid_field_type():
    """Field spaces reject invalid field type metadata."""
    schema = build_simulation_schema(_settings())

    with pytest.raises(ValueError, match="field_type"):
        field_spaces_from_bases({"bad": schema.sh_basis}, {"bad": "vector"})


def test_field_spaces_from_bases_requires_matching_keys():
    """Basis and field-type schemas must describe the same groups."""
    schema = build_simulation_schema(_settings())

    with pytest.raises(ValueError, match="same keys"):
        field_spaces_from_bases({"bad": schema.sh_basis}, {})


def test_schema_mean_free_projection_is_operational_for_cs_potential_space():
    """Surface-potential metadata applies the CS mean-free gauge."""
    schema = build_simulation_schema(_settings(horizontal_basis_kind="CS"))
    field_space = schema.output_field_spaces["dynamic"]["Phi"]
    coeffs = np.linspace(0.0, 1.0, field_space.index_length) + 5.0

    projected = field_space.project_mean_free(coeffs)

    assert projected.shape == coeffs.shape
    np.testing.assert_allclose(schema.cs_basis.scalar_mean(projected), 0.0, atol=1e-12)


def test_schema_mappings_cannot_be_mutated_after_construction():
    """Storage metadata cannot drift after construction."""
    schema = build_simulation_schema(_settings())

    with pytest.raises(TypeError):
        schema.input_variables["new"] = ("value",)
    with pytest.raises(TypeError):
        schema.input_field_spaces["boundary_jr"] = schema.output_field_spaces["dynamic"]
    with pytest.raises(TypeError):
        schema.output_field_spaces["dynamic"]["induced_Br"] = schema.input_field_spaces[
            "boundary_jr"
        ]
