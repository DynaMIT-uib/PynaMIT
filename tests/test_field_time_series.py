"""Tests for representation-aware field time series."""

import numpy as np
import pytest
import xarray as xr
from kompe import GlobalCSBasis, SHBasis
from kompe.coefficients import CoefficientSpace
from kompe.math import get_array_module

from pynamit.storage.field_time_series import TIME_TOLERANCE_SECONDS, FieldTimeSeries


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
@pytest.mark.parametrize(
    "times", [[0.0, 2.0, 4.0], [4.0, 0.0, 2.0], [0.0, 1.5e-6, 0.75e-6], [2.0, 2.0 + 0.5e-6, 2.0]]
)
def test_batch_insertion_matches_ordered_single_insertions(backend, representation, times):
    """Batches preserve gauges and tolerant replacement order."""
    space = CoefficientSpace(GlobalCSBasis(4), representation=representation, mean_free=True)
    batch = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    singles = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    xp = get_array_module()
    values = xp.arange(len(times) * space.size, dtype=float).reshape((len(times),) + space.shape)
    for series in (batch, singles):
        for time in [-1.0, 0.0, 1.0, 3.0, 5.0]:
            series.add_entry("sample", {"value": xp.ones(space.shape)}, time)
    batch.add_entries("sample", {"value": values}, times)
    for time, value in zip(times, values, strict=True):
        singles.add_entry("sample", {"value": value}, time)
    xr.testing.assert_identical(batch.datasets["sample"], singles.datasets["sample"])
    assert batch._pending_start == singles._pending_start
    assert batch._full_save_required == singles._full_save_required


def test_batch_storage_builds_metadata_and_transfers_values_once(backend, monkeypatch):
    """One variable batch needs one CPU boundary, not one per row."""
    import pynamit.storage.field_time_series as module

    space = CoefficientSpace(SHBasis(3, 2))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    values = get_array_module().ones((100, space.size))
    transfers = []
    original = module.to_numpy

    def transfer(array):
        transfers.append(array.shape)
        return original(array)

    def unexpected_concat(*args, **kwargs):
        pytest.fail("An initial coefficient batch requires no concatenation.")

    monkeypatch.setattr(module, "to_numpy", transfer)
    monkeypatch.setattr(xr, "concat", unexpected_concat)
    series.add_entries("sample", {"value": values}, np.arange(100.0))
    assert transfers == [(100, space.size)]
    assert series.datasets["sample"].sizes["time"] == 100


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
def test_batch_append_and_replace_survive_persistence(tmp_path, storage):
    """Batch writes preserve appends and changed checkpoints."""
    from pynamit.storage import ArtifactStore

    if storage == "zarr":
        pytest.importorskip("zarr")
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    store = ArtifactStore(tmp_path, preferred_dataset_storage=storage)
    series.add_entries("sample", {"value": np.zeros((2, space.size))}, [0.0, 1.0])
    series.save("sample", store)
    series.add_entries("sample", {"value": np.ones((2, space.size))}, [2.0, 3.0])
    assert series._pending_start["sample"] == 2
    assert not series._full_save_required["sample"]
    series.save("sample", store)
    series.add_entries("sample", {"value": np.full((2, space.size), 2.0)}, [1.0, 2.5])
    assert series._full_save_required["sample"]
    series.save("sample", store)
    loaded = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    loaded.load("sample", store)
    xr.testing.assert_identical(loaded.datasets["sample"], series.datasets["sample"])


def test_batch_storage_owns_values_and_rejects_incomplete_rows():
    """Caller mutations cannot silently change stored checkpoints."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    values = np.ones((2, space.size))
    series.add_entries("sample", {"value": values}, [0.0, 1.0])
    values[:] = 9.0
    np.testing.assert_array_equal(series.get_entry("sample", 0.0)["value"], np.ones(space.size))
    with pytest.raises(ValueError, match="coefficient rows"):
        series.add_entries("sample", {"value": values}, [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        series.add_entries("sample", {"value": values}, [0.0, np.nan])


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


@pytest.mark.parametrize("interpolation", [False, True])
def test_timeseries_selects_irregular_checkpoints_with_one_time_policy(interpolation):
    """Selection retains the tolerance, bounds, and mixed layouts."""
    basis = SHBasis(2, 1)
    spaces = {
        "scalar": CoefficientSpace(basis),
        "wind": CoefficientSpace(basis, representation="helmholtz"),
    }
    series = FieldTimeSeries({"sample": spaces}, {"sample": tuple(spaces)})
    times = np.array([1.0, 2.5, 8.0])
    for time in times:
        series.add_entry(
            "sample", {name: np.full(space.shape, time) for name, space in spaces.items()}, time
        )
    eps = TIME_TOLERANCE_SECONDS
    for time in [-1.0, 1.0 - eps / 2, 1.0 + eps / 2, 1.5, 2.5 - eps / 2, 6.0, 9.0]:
        selected = series.get_entry("sample", time, interpolation=interpolation)
        preceding = times[times <= time + eps]
        if preceding.size == 0:
            assert selected is None
            continue
        expected = preceding[-1]
        if interpolation and expected < time - eps and np.any(times > time + eps):
            expected = time
        for name, space in spaces.items():
            np.testing.assert_allclose(selected[name], np.full(space.size, expected))


def test_timeseries_interpolation_preserves_single_precision_values():
    """Time-axis dtype does not promote the stored field's precision."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    for time in [0.0, 2.0]:
        series.add_entry("sample", {"value": np.full(space.shape, time, dtype=np.float32)}, time)
    series.datasets["sample"] = series.datasets["sample"].assign_coords(
        time=np.array([0.0, 2.0], dtype=np.float32)
    )
    selected = series.get_entry("sample", 0.5, interpolation=True)["value"]
    assert selected.dtype == np.float32
    np.testing.assert_array_equal(selected, np.full(space.size, 0.5, dtype=np.float32))


def test_timeseries_empty_stream_has_no_entry_but_unknown_key_is_an_error():
    """A known, unsampled stream is different from a misspelled key."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    assert series.get_entry("sample", 0.0) is None
    with pytest.raises(KeyError):
        series.get_entry("unknown", 0.0)
    series.add_entry("sample", {"value": np.ones(space.shape)}, 0.0)
    series.datasets["sample"] = series.datasets["sample"].isel(time=slice(0, 0))
    assert series.get_entry("sample", 0.0, interpolation=True) is None


def test_timeseries_requires_field_space_and_name_only_variables():
    """Time-series schema keeps field types in CoefficientSpace only."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar")

    with pytest.raises(TypeError, match="field types belong in CoefficientSpace"):
        FieldTimeSeries({"sample": field_space}, {"sample": {"value": "scalar"}})

    with pytest.raises(ValueError, match="same keys"):
        FieldTimeSeries({"sample": field_space}, {"other": ("value",)})
