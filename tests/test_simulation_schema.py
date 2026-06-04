"""Tests for simulation storage schema construction."""

import numpy as np
import pytest
import xarray as xr

from pynamit.simulation.schema import build_simulation_schema, field_spaces_from_bases


def _settings(**attrs):
    defaults = {
        "Nmax": 3,
        "Mmax": 2,
        "Ncs": 4,
        "vector_jr": 1,
        "vector_Br": 1,
        "vector_conductance": 1,
        "vector_u": 1,
    }
    defaults.update(attrs)
    return xr.Dataset(attrs=defaults)


def test_sh_schema_uses_mean_free_sh_inputs_and_outputs():
    """SH mode keeps the established mean-free SH storage choices."""
    schema = build_simulation_schema(_settings(), "SH")

    assert schema.horizontal_basis is schema.sh_basis_mean_free
    assert schema.radial_continuation_basis is schema.horizontal_basis
    assert schema.input_field_spaces["jr"].representation is schema.sh_basis_mean_free
    assert schema.input_field_spaces["Br"].representation is schema.sh_basis_mean_free
    assert schema.input_field_spaces["u"].representation is schema.sh_basis_mean_free
    assert schema.input_field_spaces["conductance"].representation is schema.sh_basis
    assert schema.output_field_spaces["state"].representation is schema.horizontal_basis

    assert schema.input_field_spaces["jr"].mean_free
    assert schema.input_field_spaces["Br"].mean_free
    assert schema.input_field_spaces["u"].mean_free
    assert not schema.input_field_spaces["conductance"].mean_free
    assert schema.output_field_spaces["state"].mean_free


def test_cs_schema_uses_full_length_storage_with_mean_free_intent():
    """CS mode stores full grid coefficients with zero-mean intent."""
    schema = build_simulation_schema(_settings(), "cs")

    assert schema.horizontal_basis is schema.cs_basis
    assert schema.radial_continuation_basis is schema.sh_basis_mean_free
    assert all(
        space.representation is schema.cs_basis
        for space in schema.input_field_spaces.values()
    )
    assert all(
        space.representation is schema.cs_basis
        for space in schema.output_field_spaces.values()
    )
    assert all(basis is schema.cs_basis for basis in schema.interpolation_bases.values())

    state_space = schema.output_field_spaces["state"]
    assert state_space.mean_free
    assert state_space.index_length == schema.cs_basis.index_length
    assert state_space.coefficient_length == schema.cs_basis.index_length


def test_schema_respects_vector_input_flags_for_sh_projection_basis():
    """Projection bases stay independent from storage bases."""
    schema = build_simulation_schema(
        _settings(vector_jr=0, vector_Br=0, vector_conductance=0, vector_u=0),
        "SH",
    )

    assert schema.input_field_spaces["jr"].representation is schema.sh_basis_mean_free
    assert schema.input_field_spaces["conductance"].representation is schema.sh_basis
    assert all(basis is schema.cs_basis for basis in schema.interpolation_bases.values())


def test_sh_schema_can_store_conductance_on_grid_without_projection():
    """No-projection conductance keeps SH state storage."""
    schema = build_simulation_schema(_settings(project_conductance=0), "SH")

    assert schema.horizontal_basis is schema.sh_basis_mean_free
    assert schema.input_field_spaces["conductance"].representation is schema.cs_basis
    assert schema.interpolation_bases["conductance"] is schema.cs_basis
    assert not schema.input_field_spaces["conductance"].mean_free


def test_field_spaces_from_bases_rejects_invalid_field_type():
    """Field spaces reject invalid field type metadata."""
    schema = build_simulation_schema(_settings(), "SH")

    with pytest.raises(ValueError, match="field_type"):
        field_spaces_from_bases(
            {"bad": schema.sh_basis},
            {"bad": "vector"},
        )


def test_field_spaces_from_bases_requires_matching_keys():
    """Basis and field-type schemas must describe the same groups."""
    schema = build_simulation_schema(_settings(), "SH")

    with pytest.raises(ValueError, match="same keys"):
        field_spaces_from_bases({"bad": schema.sh_basis}, {})


def test_schema_mean_free_projection_is_operational_for_cs_state_space():
    """Schema FieldSpace metadata can project CS coefficients."""
    schema = build_simulation_schema(_settings(), "CS")
    field_space = schema.output_field_spaces["state"]
    coeffs = np.linspace(0.0, 1.0, field_space.index_length) + 5.0

    projected = field_space.project_mean_free(coeffs)

    assert projected.shape == coeffs.shape
    np.testing.assert_allclose(schema.cs_basis.scalar_mean(projected), 0.0, atol=1e-12)
