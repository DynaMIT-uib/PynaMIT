"""Tests for representation-aware field time series."""

import numpy as np
import pytest
from kompe import GlobalCSBasis, SHBasis
from kompe.coefficients import CoefficientSpace

from pynamit.storage.field_time_series import TIME_TOLERANCE_SECONDS, FieldTimeSeries


def test_timeseries_exposes_field_space_and_projects_mean_free_cs_coefficients():
    """Time-series storage honors CoefficientSpace metadata."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar", mean_free=True)
    timeseries = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    values = np.linspace(0.0, 1.0, basis.coefficient_count) + 2.0

    timeseries.add_entry("sample", {"value": values}, time=0.0)

    assert timeseries.get_field_space("sample") is field_space
    assert timeseries.get_data_var_name("sample", "value") == "CS_value"
    stored = timeseries.get_entry("sample", 0.0)["value"]
    np.testing.assert_allclose(basis.scalar_mean(stored), 0.0, atol=1e-12)


def test_timeseries_replaces_near_equal_floating_time():
    """Replace checkpoints using the declared time tolerance."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    timeseries = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    first = np.zeros(field_space.shape)
    replacement = np.ones(field_space.shape)

    timeseries.add_entry("sample", {"value": first}, time=1.0)
    timeseries.add_entry("sample", {"value": replacement}, time=1.0 + 0.5e-6)

    assert timeseries.datasets["sample"].sizes["time"] == 1
    np.testing.assert_allclose(timeseries.get_entry("sample", 1.0)["value"], replacement)


@pytest.mark.parametrize("time", [np.nan, np.inf, [0.0], True])
def test_timeseries_rejects_invalid_entry_time(time):
    """Stored simulation times must be finite numeric scalars."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    timeseries = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})

    with pytest.raises(ValueError, match="time value"):
        timeseries.add_entry("sample", {"value": np.zeros(field_space.shape)}, time=time)


def test_timeseries_does_not_interpolate_across_tolerant_time_match():
    """A near checkpoint match selects that checkpoint exactly."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    timeseries = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    first = np.full(field_space.shape, 10.0)
    second = np.full(field_space.shape, 20.0)
    timeseries.add_entry("sample", {"value": first}, time=1.0)
    timeseries.add_entry("sample", {"value": second}, time=2.0)

    selected = timeseries.get_entry(
        "sample", 1.0 - 0.5 * TIME_TOLERANCE_SECONDS, interpolation=True
    )

    np.testing.assert_array_equal(selected["value"], first)


def test_timeseries_rejects_loaded_coefficient_index_mismatch():
    """Restart artifacts preserve coefficient identity and length."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    source = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    source.add_entry("sample", {"value": np.zeros(field_space.shape)}, time=0.0)
    persisted = source.datasets["sample"].reset_index("i")
    first_index_name = field_space.index_names[0]
    persisted = persisted.assign_coords(
        {first_index_name: ("i", persisted[first_index_name].values[::-1])}
    )

    class _LoadedDataset:
        @staticmethod
        def get_dataset_storage_kind(_key):
            return "netcdf"

        @staticmethod
        def load_dataset(_key):
            return persisted

    restored = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    with pytest.raises(ValueError, match="coefficient index"):
        restored.load("sample", _LoadedDataset())


def test_timeseries_restores_coefficient_multiindex_in_memory():
    """Loaded series recover their in-memory coefficient index."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    source = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    source.add_entry("sample", {"value": np.zeros(field_space.shape)}, time=0.0)
    persisted = source.datasets["sample"].reset_index("i")

    class _LoadedDataset:
        @staticmethod
        def get_dataset_storage_kind(_key):
            return "netcdf"

        @staticmethod
        def load_dataset(_key):
            return persisted

    restored = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    restored.load("sample", _LoadedDataset())

    dataset = restored.datasets["sample"]
    assert "i" in dataset.indexes
    assert tuple(dataset.indexes["i"].names) == tuple(field_space.index_names)
    assert dataset.reset_index("i").equals(persisted)


def test_tangential_timeseries_labels_components_and_physical_metadata():
    """Tangential data identify both Helmholtz coefficient blocks."""
    basis = SHBasis(2, 1, mean_free=True)
    field_space = CoefficientSpace(basis, representation="helmholtz")
    timeseries = FieldTimeSeries(
        {"wind": field_space},
        {"wind": ("u",)},
        variable_attrs={"wind": {"u": {"units": "m s-1", "long_name": "neutral wind velocity"}}},
        time_origin="2020-01-01 00:00:00",
    )
    timeseries.add_entry("wind", {"u": np.zeros(field_space.shape)}, time=3.0)

    dataset = timeseries.datasets["wind"]
    np.testing.assert_array_equal(
        dataset.component.values,
        np.repeat(np.array([0, 1], dtype=np.int8), field_space.coefficient_count),
    )
    assert dataset.component.attrs["long_name"] == "Helmholtz potential coefficient component"
    assert (
        dataset.component.attrs["flag_meanings"] == "curl_free_potential divergence_free_potential"
    )
    assert dataset.time.attrs == {
        "units": "s",
        "long_name": "simulation time since t0",
        "time_origin": "2020-01-01 00:00:00",
    }
    assert dataset["SH_u"].attrs["units"] == "m s-1"
    assert dataset["SH_u"].attrs["field_type"] == "tangential"


def test_tangential_timeseries_adds_component_labels_when_loading_older_data():
    """Older artifacts may omit the auxiliary component label."""
    basis = SHBasis(2, 1, mean_free=True)
    field_space = CoefficientSpace(basis, representation="helmholtz")
    source = FieldTimeSeries({"wind": field_space}, {"wind": ("u",)})
    source.add_entry("wind", {"u": np.zeros(field_space.shape)}, time=0.0)
    persisted = source.datasets["wind"].reset_index("i").drop_vars("component")

    class _LoadedDataset:
        @staticmethod
        def get_dataset_storage_kind(_key):
            return "netcdf"

        @staticmethod
        def load_dataset(_key):
            return persisted

    restored = FieldTimeSeries({"wind": field_space}, {"wind": ("u",)})
    restored.load("wind", _LoadedDataset())

    assert "component" in restored.datasets["wind"].coords


def test_timeseries_selection_is_stateless():
    """Repeated readers always receive the selected coefficients."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar")
    timeseries = FieldTimeSeries(
        {"first": field_space, "second": field_space}, {"first": ("value",), "second": ("value",)}
    )
    values = np.zeros(basis.coefficient_count)
    timeseries.add_entry("first", {"value": values}, time=0.0)
    timeseries.add_entry("second", {"value": values}, time=0.0)

    for key in ("first", "second", "first", "second"):
        np.testing.assert_array_equal(timeseries.get_entry(key, 0.0)["value"], values)


def test_timeseries_selection_preserves_small_changes_and_stored_values():
    """Selected values do not overwrite the stored history."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    timeseries = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    first = np.ones(field_space.shape)
    second = first + 1e-7
    timeseries.add_entry("sample", {"value": first}, time=0.0)
    timeseries.add_entry("sample", {"value": second}, time=2.0)

    selected = timeseries.get_entry("sample", 0.0)
    selected["value"][:] = 10.0
    np.testing.assert_array_equal(timeseries.get_entry("sample", 0.0)["value"], first)

    changed = timeseries.get_entry("sample", 1.0, interpolation=True)
    assert changed is not None
    np.testing.assert_allclose(changed["value"], first + 0.5e-7, rtol=0.0, atol=1e-15)


def test_timeseries_requires_field_space_and_name_only_variables():
    """Time-series schema keeps field types in CoefficientSpace only."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar")

    with pytest.raises(TypeError, match="field types belong in CoefficientSpace"):
        FieldTimeSeries({"sample": field_space}, {"sample": {"value": "scalar"}})

    with pytest.raises(ValueError, match="same keys"):
        FieldTimeSeries({"sample": field_space}, {"other": ("value",)})
