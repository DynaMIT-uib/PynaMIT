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
    assert schema.input_storage_bases["jr"] is schema.sh_basis_mean_free
    assert schema.input_storage_bases["Br"] is schema.sh_basis_mean_free
    assert schema.input_storage_bases["u"] is schema.sh_basis_mean_free
    assert schema.input_storage_bases["conductance"] is schema.sh_basis
    assert schema.output_storage_bases["state"] is schema.horizontal_basis

    assert schema.input_field_spaces["jr"].mean_free
    assert schema.input_field_spaces["Br"].mean_free
    assert schema.input_field_spaces["u"].mean_free
    assert not schema.input_field_spaces["conductance"].mean_free
    assert schema.output_field_spaces["state"].mean_free


def test_cs_schema_uses_full_length_storage_with_mean_free_intent():
    """CS mode stores full grid coefficients while recording zero-mean intent."""
    schema = build_simulation_schema(_settings(), "cs")

    assert schema.horizontal_basis is schema.cs_basis
    assert schema.radial_continuation_basis is schema.sh_basis_mean_free
    assert all(basis is schema.cs_basis for basis in schema.input_storage_bases.values())
    assert all(basis is schema.cs_basis for basis in schema.output_storage_bases.values())
    assert all(basis is schema.cs_basis for basis in schema.interpolation_bases.values())

    state_space = schema.output_field_spaces["state"]
    assert state_space.mean_free
    assert state_space.index_length == schema.cs_basis.index_length
    assert state_space.coefficient_length == schema.cs_basis.index_length


def test_schema_respects_vector_input_flags_for_sh_projection_basis():
    """Projection bases stay independent from persisted storage bases."""
    schema = build_simulation_schema(
        _settings(vector_jr=0, vector_Br=0, vector_conductance=0, vector_u=0),
        "SH",
    )

    assert schema.input_storage_bases["jr"] is schema.sh_basis_mean_free
    assert schema.input_storage_bases["conductance"] is schema.sh_basis
    assert all(basis is schema.cs_basis for basis in schema.interpolation_bases.values())


def test_field_spaces_from_bases_rejects_mixed_field_types():
    """One stored time-series key cannot mix scalar and tangential variables."""
    schema = build_simulation_schema(_settings(), "SH")

    with pytest.raises(ValueError, match="Mixed scalar and tangential"):
        field_spaces_from_bases(
            {"bad": schema.sh_basis},
            {"bad": {"a": "scalar", "b": "tangential"}},
        )


def test_schema_mean_free_projection_is_operational_for_cs_state_space():
    """FieldSpace metadata from the schema can project CS coefficients."""
    schema = build_simulation_schema(_settings(), "CS")
    field_space = schema.output_field_spaces["state"]
    coeffs = np.linspace(0.0, 1.0, field_space.index_length) + 5.0

    projected = field_space.project_mean_free(coeffs)

    assert projected.shape == coeffs.shape
    np.testing.assert_allclose(schema.cs_basis.scalar_mean(projected), 0.0, atol=1e-12)
