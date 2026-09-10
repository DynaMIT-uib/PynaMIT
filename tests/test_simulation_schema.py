"""Tests for simulation storage schema construction."""

import numpy as np
import pytest

from pynamit.simulation.config import (
    SimulationConfig,
    normalize_horizontal_basis_kind,
    normalize_input_remapping,
    resolve_input_remapping,
)
from pynamit.simulation.geometry import SimulationGeometry
from pynamit.simulation.schema import INPUT_VARIABLES, build_simulation_schema


def _geometry_and_schema(config):
    geometry = SimulationGeometry.from_config(config)
    return geometry, build_simulation_schema(geometry, config)


def _settings(**attrs):
    defaults = {"Nmax": 3, "Mmax": 2, "Ncs": 4}
    defaults.update(attrs)
    return SimulationConfig(**defaults)


def test_horizontal_basis_kind_is_simulation_policy():
    """Normalize horizontal basis choices within simulation policy."""
    assert normalize_horizontal_basis_kind(" sh ") == "SH"
    assert normalize_horizontal_basis_kind("cs") == "CS"
    with pytest.raises(ValueError, match="horizontal_basis_kind"):
        normalize_horizontal_basis_kind("grid")


def test_remapping_is_input_policy():
    """Normalize sample-remapping choices within simulation policy."""
    assert normalize_input_remapping(" direct ", name="boundary_jr_remapping") == "direct"
    assert normalize_input_remapping("cs", name="u_remapping") == "CS"
    with pytest.raises(ValueError, match="boundary_jr_remapping"):
        normalize_input_remapping("grid", name="boundary_jr_remapping")


def test_remapping_settings_resolve_defaults_and_inheritance():
    """Remapping settings share one normalization path."""
    settings = {"u_remapping": "CS"}

    resolved = resolve_input_remapping(settings, "SH")

    assert resolved == {
        "boundary_jr_remapping": "direct",
        "boundary_Br_remapping": "direct",
        "u_remapping": "CS",
        "E_neutral_wind_remapping": "direct",
        "Q_eff_remapping": "CS",
    }


def test_remapping_settings_require_model_nodes_in_cs_mode():
    """A CS horizontal basis requires matching CS input settings."""
    with pytest.raises(ValueError, match="boundary_jr_remapping"):
        resolve_input_remapping({"boundary_jr_remapping": "direct"}, "CS")


def test_sh_schema_uses_mean_free_sh_inputs_and_outputs():
    """SH mode keeps the established mean-free SH storage choices."""
    geometry, schema = _geometry_and_schema(_settings())

    assert geometry.horizontal_basis is geometry.poloidal_basis
    assert geometry.solid_harmonics.basis is geometry.horizontal_basis
    assert schema.input_field_spaces["boundary_jr"].basis is geometry.poloidal_basis
    assert schema.input_field_spaces["boundary_Br"].basis is geometry.poloidal_basis
    assert schema.input_field_spaces["u"].basis is geometry.poloidal_basis
    assert schema.input_field_spaces["Q_eff"].basis is geometry.poloidal_basis
    assert schema.input_field_spaces["conductance"].basis is geometry.sh_basis
    assert all(
        space.basis is geometry.horizontal_basis
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
    geometry, schema = _geometry_and_schema(_settings(horizontal_basis_kind="cs"))

    assert geometry.horizontal_basis is geometry.cs_basis
    assert geometry.solid_harmonics.basis is geometry.poloidal_basis
    assert schema.input_field_spaces["boundary_Br"].basis is geometry.poloidal_basis
    assert all(
        space.basis is geometry.cs_basis
        for key, space in schema.input_field_spaces.items()
        if key != "boundary_Br"
    )

    output_spaces = schema.output_field_spaces["dynamic"]
    assert output_spaces["induced_Br"].basis is geometry.poloidal_basis
    assert all(
        output_spaces[name].basis is geometry.cs_basis for name in ("boundary_jr", "Phi", "W")
    )
    assert output_spaces["induced_Br"].mean_free
    assert not output_spaces["boundary_jr"].mean_free
    assert output_spaces["Phi"].mean_free
    assert output_spaces["W"].mean_free


def test_remapping_does_not_change_coefficient_spaces():
    """Remapping leaves coefficient spaces fixed."""
    geometry, schema = _geometry_and_schema(
        _settings(
            boundary_jr_remapping="CS",
            boundary_Br_remapping="CS",
            conductance_basis="CS",
            u_remapping="CS",
            Q_eff_remapping="CS",
            E_neutral_wind_remapping="CS",
        )
    )

    assert schema.input_field_spaces["boundary_jr"].basis is geometry.poloidal_basis
    assert schema.input_field_spaces["conductance"].basis is geometry.cs_basis


def test_sh_schema_can_store_conductance_on_cs_grid():
    """CS conductance storage keeps the SH horizontal basis."""
    geometry, schema = _geometry_and_schema(_settings(conductance_basis="CS"))

    assert geometry.horizontal_basis is geometry.poloidal_basis
    assert schema.input_field_spaces["conductance"].basis is geometry.cs_basis
    assert not schema.input_field_spaces["conductance"].mean_free


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
def test_schema_creates_independent_input_series_with_physical_metadata(horizontal_basis_kind):
    """Input streams keep their units and UTC origin in either basis."""
    config = _settings(horizontal_basis_kind=horizontal_basis_kind)
    geometry, schema = _geometry_and_schema(config)
    series = schema.create_input_series(time_origin=config.t0)
    other = schema.create_input_series(time_origin=config.t0)
    expected_units = {
        "boundary_jr": "A m-2",
        "boundary_Br": "T",
        "conductance": "1",
        "u": "m s-1",
        "Q_eff": "A m-1",
        "E_neutral_wind": "V m-1",
    }

    for key, units in expected_units.items():
        space = schema.input_field_spaces[key]
        data = {name: np.zeros(space.shape) for name in schema.input_variables[key]}
        series.add_entry(key, data, time=3.0)
        dataset = series.datasets[key]
        row = series.get_entry(key, 3.0)
        assert dataset.time.attrs["time_origin"] == config.t0
        assert dataset.time.attrs["units"] == "s"
        for name in data:
            variable = series.get_data_var_name(key, name)
            assert dataset[variable].attrs["units"] == units
            np.testing.assert_array_equal(row[name], data[name])

    assert other.datasets == {}


@pytest.mark.parametrize("horizontal_basis_kind", ["SH", "CS"])
def test_schema_creates_independent_output_series_with_physical_metadata(horizontal_basis_kind):
    """Output streams keep units and mixed SH/CS coefficient layouts."""
    config = _settings(horizontal_basis_kind=horizontal_basis_kind)
    geometry, schema = _geometry_and_schema(config)
    series = schema.create_output_series(time_origin=config.t0)
    other = schema.create_output_series(time_origin=config.t0)
    expected_units = {"induced_Br": "T", "boundary_jr": "A m-2", "Phi": "V m-1", "W": "V m-1"}

    for key, spaces in schema.output_field_spaces.items():
        data = {name: np.zeros(space.shape) for name, space in spaces.items()}
        series.add_entry(key, data, time=3.0)
        dataset = series.datasets[key]
        row = series.get_entry(key, 3.0)
        assert dataset.time.attrs["time_origin"] == config.t0
        assert dataset.time.attrs["units"] == "s"
        for name, units in expected_units.items():
            variable = series.get_data_var_name(key, name)
            assert dataset[variable].attrs["units"] == units
            np.testing.assert_array_equal(row[name], data[name])

    assert other.datasets == {}


def test_schema_mean_free_projection_is_operational_for_cs_potential_space():
    """Surface-potential metadata applies the CS mean-free gauge."""
    geometry, schema = _geometry_and_schema(_settings(horizontal_basis_kind="CS"))
    field_space = schema.output_field_spaces["dynamic"]["Phi"]
    coeffs = np.linspace(0.0, 1.0, field_space.coefficient_count) + 5.0

    projected = field_space.project_mean_free(coeffs)

    assert projected.shape == coeffs.shape
    np.testing.assert_allclose(geometry.cs_basis.scalar_mean(projected), 0.0, atol=1e-12)


def test_schema_mappings_are_ordinary_independent_dictionaries():
    """Keep storage metadata inspectable without aliasing constants."""
    geometry, schema = _geometry_and_schema(_settings())

    schema.input_variables["new"] = ("value",)

    assert isinstance(schema.input_variables, dict)
    assert isinstance(schema.output_field_spaces["dynamic"], dict)
    assert "new" not in INPUT_VARIABLES
