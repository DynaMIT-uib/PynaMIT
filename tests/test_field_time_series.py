"""Tests for representation-aware field time series."""

import numpy as np
import pytest
import xarray as xr
from kompe import GlobalCSBasis, SHBasis
from kompe.coefficients import CoefficientSpace
from kompe.math import get_array_module

from pynamit.storage import ArtifactStore
from pynamit.storage.field_time_series import FieldTimeSeries, time_roundoff


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
@pytest.mark.parametrize("interpolation", [False, True])
def test_time_block_selection_retains_shapes_order_onsets_and_roundoff(
    representation, interpolation
):
    """Batches preserve causal sampling and coefficient axes."""
    space = CoefficientSpace(SHBasis(2, 1), representation=representation)
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    times = np.array([0.3, 1.0, 2.0, 2.000001])
    coefficients = np.arange(times.size * space.size).reshape((times.size,) + space.shape)
    series.add_entries("sample", {"value": coefficients}, times)
    queries = np.array([3.0, 0.3, 3 * 0.1, 0.7, 2.0000005, 1.0, 0.7])
    block = series.get_entry("sample", queries, interpolation)["value"]
    assert block.shape == space.shape + (queries.size,)
    expected = np.stack(
        [series.get_entry("sample", time, interpolation)["value"] for time in queries], axis=-1
    )
    np.testing.assert_allclose(block, expected)
    # Manufactured interpolation/holding check, not just scalar parity.
    expected_at_point = (
        coefficients[0] + (4 / 7) * (coefficients[1] - coefficients[0])
        if interpolation
        else coefficients[0]
    )
    np.testing.assert_allclose(block[..., 3], expected_at_point)
    np.testing.assert_array_equal(block[..., 1], coefficients[0])
    np.testing.assert_array_equal(block[..., 2], coefficients[0])
    assert series.get_entry("sample", [0.0, 0.7], interpolation) is None
    filled = series.get_entry("sample", [0.0, 0.7], interpolation, fill_value=0.0)["value"]
    np.testing.assert_array_equal(filled[..., 0], 0)
    np.testing.assert_allclose(filled[..., 1], expected_at_point)
    block[...] = -100
    np.testing.assert_array_equal(series.get_entry("sample", 0.3)["value"], coefficients[0])


@pytest.mark.parametrize("times", [[], [[1]], [np.nan], [np.inf], [True], ["1"]])
def test_time_block_selection_rejects_invalid_query_times(times):
    """Time-array validation happens at the selection boundary."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    with pytest.raises(ValueError, match="time"):
        series.get_entry("sample", times)


def test_monotonic_appends_copy_history_only_when_capacity_grows(monkeypatch):
    """Small batches have linear coefficient-copying cost."""
    space = CoefficientSpace(SHBasis(3, 2))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    copied_rows = []
    original = np.copyto

    def copyto(target, source, **kwargs):
        if target.ndim == 2 and target.shape[1] == space.size:
            copied_rows.append(target.shape[0])
        return original(target, source, **kwargs)

    monkeypatch.setattr(np, "copyto", copyto)

    def no_history_concatenation(*args, **kwargs):
        pytest.fail("A monotonic append must not concatenate the old history.")

    monkeypatch.setattr(xr, "concat", no_history_concatenation)
    for start in range(0, 1000, 10):
        times = np.arange(start, start + 10, dtype=float)
        values = np.broadcast_to(times[:, None], (10, space.size))
        series.add_entries("sample", {"value": values}, times)
    assert 0 < sum(copied_rows) < 2000
    assert len(copied_rows) < 10
    np.testing.assert_array_equal(series.datasets["sample"].time, np.arange(1000))
    np.testing.assert_array_equal(
        series.datasets["sample"].SH_value,
        np.broadcast_to(np.arange(1000)[:, None], (1000, space.size)),
    )


@pytest.mark.parametrize("edit", ["in_place", "variable", "dataset", "replacement"])
def test_buffered_appends_preserve_live_edits_and_independent_snapshots(edit):
    """The exposed dataset, not an old buffer, remains authoritative."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    for time in range(5):
        series.add_entry("sample", {"value": np.full(space.shape, float(time))}, time)
    dataset = series.datasets["sample"]
    snapshot = dataset.copy(deep=True)
    if edit == "in_place":
        dataset.SH_value.values[0] = 9
    elif edit == "variable":
        values = dataset.SH_value.values.copy()
        values[0] = 9
        dataset["SH_value"] = (dataset.SH_value.dims, values)
    elif edit == "dataset":
        dataset = dataset.copy(deep=True)
        dataset.SH_value.values[0] = 9
        series.datasets["sample"] = dataset
    else:
        series.add_entry("sample", {"value": np.full(space.shape, 9.0)}, 0.0)
    series.datasets["sample"].attrs["note"] = "edited interactively"
    series.add_entry("sample", {"value": np.full(space.shape, 5.0)}, 5.0)
    np.testing.assert_array_equal(series.get_entry("sample", 0)["value"], 9)
    np.testing.assert_array_equal(series.get_entry("sample", 5)["value"], 5)
    np.testing.assert_array_equal(snapshot.SH_value.values[0], 0)
    assert series.datasets["sample"].attrs["note"] == "edited interactively"


def test_appending_requires_the_original_coefficient_order():
    """An edited coordinate index must not silently reorder new rows."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    series.add_entry("sample", {"value": np.ones(space.shape)}, 0.0)
    edited = series.datasets["sample"].isel(i=slice(None, None, -1))
    series.datasets["sample"] = edited
    with pytest.raises(ValueError, match="Coefficient indexes"):
        series.add_entry("sample", {"value": np.ones(space.shape)}, 1.0)
    assert series.datasets["sample"] is edited


def test_append_buffer_promotes_values_without_losing_complex_parts():
    """Appending higher-precision values promotes the whole history."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    for time in range(5):
        series.add_entry("sample", {"value": np.full(space.shape, time, dtype=np.float32)}, time)
    series.add_entry("sample", {"value": np.full(space.shape, 2 + 3j)}, 5)
    np.testing.assert_array_equal(series.get_entry("sample", 4)["value"], 4)
    np.testing.assert_array_equal(series.get_entry("sample", 5)["value"], 2 + 3j)
    assert series.datasets["sample"].SH_value.dtype == np.complex128


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
    actual, expected = batch.datasets["sample"], singles.datasets["sample"]
    # Batched JAX reductions can differ by a few rounding bits.
    # Time replacement, coordinates, and metadata must still be exact.
    atol = 8 * np.finfo(float).eps * float(xp.max(xp.abs(values)))
    xr.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
    xr.testing.assert_identical(
        actual.drop_vars(list(actual.data_vars)), expected.drop_vars(list(expected.data_vars))
    )
    assert actual.CS_value.attrs == expected.CS_value.attrs
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

    normalizations = []
    original_projection = CoefficientSpace.project_mean_free

    def project(space, array, **kwargs):
        normalizations.append(array.shape)
        return original_projection(space, array, **kwargs)

    monkeypatch.setattr(module, "to_numpy", transfer)
    monkeypatch.setattr(CoefficientSpace, "project_mean_free", project)
    monkeypatch.setattr(xr, "concat", unexpected_concat)
    monkeypatch.setattr(
        xr.Coordinates,
        "from_pandas_multiindex",
        lambda *args, **kwargs: pytest.fail("Coefficient metadata is already constructed."),
    )
    series.add_entries("sample", {"value": values}, np.arange(100.0))
    assert transfers == [(100, space.size)]
    assert normalizations == [(space.size, 100)]
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


def test_chronological_append_constructs_one_dataset(monkeypatch):
    """Append owned rows without an intermediate xarray dataset."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    values = np.ones((2, space.size))
    series.add_entries("sample", {"value": values}, [0.0, 1.0])
    calls = []
    original = xr.Dataset.__init__

    def initialize(self, *args, **kwargs):
        calls.append(None)
        original(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "__init__", initialize)
    series.add_entries("sample", {"value": 2 * values}, [2.0, 3.0])
    assert len(calls) == 1
    np.testing.assert_array_equal(series.get_entry("sample", 3.0)["value"], 2 * values[0])


@pytest.mark.parametrize("representation", ["scalar", "helmholtz"])
def test_batch_storage_accepts_grid_shaped_coefficients(backend, representation):
    """CS rows may retain their native face and cell axes."""
    space = CoefficientSpace(GlobalCSBasis(4), representation=representation, mean_free=True)
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    values = np.random.default_rng(429).normal(size=(3,) + space.shape)
    grid_shape = (3, 6, 4, 4) if representation == "scalar" else (3, 2, 6, 4, 4)
    series.add_entries("sample", {"value": values.reshape(grid_shape)}, [0.0, 1.0, 2.0])
    means = np.einsum("...n,n->...", values, space.basis.scalar_mean_weights)
    expected = (values - means[..., None]).reshape(3, space.size)
    np.testing.assert_allclose(series.datasets["sample"].CS_value, expected, atol=1e-14)


@pytest.mark.parametrize("storage", ["netcdf", "zarr"])
@pytest.mark.parametrize("append", [False, True])
def test_saving_to_another_store_writes_the_complete_series(tmp_path, storage, append):
    """Clean/append bookkeeping belongs to the artifact it describes."""
    from pynamit.storage import ArtifactStore

    if storage == "zarr":
        pytest.importorskip("zarr")
    space = CoefficientSpace(SHBasis(2, 1))
    source = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    unrelated = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    original = ArtifactStore(tmp_path / "original", preferred_dataset_storage=storage)
    destination = ArtifactStore(tmp_path / "copy", preferred_dataset_storage=storage)
    source.add_entries("sample", {"value": np.ones((2, space.size))}, [0.0, 1.0])
    source.save("sample", original)
    source.load("sample", original)
    unrelated.add_entry("sample", {"value": 9 * np.ones(space.shape)}, 10.0)
    unrelated.save("sample", destination)
    if append:
        source.add_entry("sample", {"value": 2 * np.ones(space.shape)}, 2.0)

    source.save("sample", destination)
    restored = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    restored.load("sample", destination)
    xr.testing.assert_identical(restored.datasets["sample"], source.datasets["sample"])

    # Returning to the original must persist the complete state too.
    source.save("sample", original)
    restored.load("sample", original)
    xr.testing.assert_identical(restored.datasets["sample"], source.datasets["sample"])


def test_copying_to_a_new_store_uses_its_requested_format(tmp_path):
    """A new destination chooses its own format, not the source's."""
    from pynamit.storage import ArtifactStore

    pytest.importorskip("zarr")
    space = CoefficientSpace(SHBasis(2, 1))
    source = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    original = ArtifactStore(tmp_path / "original", preferred_dataset_storage="zarr")
    destination = ArtifactStore(tmp_path / "copy", preferred_dataset_storage="netcdf")
    source.add_entry("sample", {"value": np.ones(space.shape)}, 0.0)
    source.save("sample", original)

    source.save("sample", destination)

    assert destination.get_dataset_storage_kind("sample") == "netcdf"


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
    timeseries.add_entry("sample", {"value": replacement}, time=np.nextafter(1.0, 2.0))

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

    selected = timeseries.get_entry("sample", 1.0 - 0.5 * time_roundoff(1.0), interpolation=True)

    np.testing.assert_array_equal(selected["value"], first)


def test_timeseries_rejects_loaded_coefficient_index_mismatch(tmp_path, monkeypatch):
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

    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "load_dataset", lambda _: persisted)
    restored = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    with pytest.raises(ValueError, match="coefficient index"):
        restored.load("sample", store)


def test_timeseries_restores_coefficient_multiindex_in_memory(tmp_path, monkeypatch):
    """Loaded series recover their in-memory coefficient index."""
    basis = SHBasis(2, 1)
    field_space = CoefficientSpace(basis)
    source = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    source.add_entry("sample", {"value": np.zeros(field_space.shape)}, time=0.0)
    persisted = source.datasets["sample"].reset_index("i")

    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "load_dataset", lambda _: persisted)
    restored = FieldTimeSeries({"sample": field_space}, {"sample": ("value",)})
    restored.load("sample", store)

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


def test_tangential_timeseries_adds_component_labels_when_loading_older_data(
    tmp_path, monkeypatch
):
    """Older artifacts may omit the auxiliary component label."""
    basis = SHBasis(2, 1, mean_free=True)
    field_space = CoefficientSpace(basis, representation="helmholtz")
    source = FieldTimeSeries({"wind": field_space}, {"wind": ("u",)})
    source.add_entry("wind", {"u": np.zeros(field_space.shape)}, time=0.0)
    persisted = source.datasets["wind"].reset_index("i").drop_vars("component")

    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "load_dataset", lambda _: persisted)
    restored = FieldTimeSeries({"wind": field_space}, {"wind": ("u",)})
    restored.load("wind", store)

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
    eps = time_roundoff(1.0)
    for time in [-1.0, 1.0 - eps / 2, 1.0 + eps / 2, 1.5, 2.5 - eps / 2, 6.0, 9.0]:
        selected = series.get_entry("sample", time, interpolation=interpolation)
        preceding = times[times <= time + time_roundoff(time)]
        if preceding.size == 0:
            assert selected is None
            continue
        expected = preceding[-1]
        if (
            interpolation
            and expected < time - time_roundoff(time)
            and np.any(times > time + time_roundoff(time))
        ):
            expected = time
        for name, space in spaces.items():
            np.testing.assert_allclose(selected[name], np.full(space.shape, expected))


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
    batched = series.get_entry("sample", [0.5, 1.0], interpolation=True)["value"]
    assert batched.dtype == np.float32
    np.testing.assert_allclose(batched[..., 0], selected)
    assert selected.dtype == np.float32
    np.testing.assert_array_equal(selected, np.full(space.size, 0.5, dtype=np.float32))


@pytest.mark.parametrize("interpolation", [False, True])
def test_selected_variables_retain_shapes_without_reading_other_coefficients(
    monkeypatch, interpolation
):
    """Subset queries load only the requested coefficients."""
    basis = SHBasis(2, 1)
    spaces = {
        "scalar": CoefficientSpace(basis),
        "wind": CoefficientSpace(basis, representation="helmholtz"),
    }
    series = FieldTimeSeries({"sample": spaces}, {"sample": tuple(spaces)})
    for time in (0.0, 1.0):
        series.add_entry(
            "sample", {name: np.full(space.shape, time) for name, space in spaces.items()}, time
        )
    original = xr.Variable.values.fget
    reads = []

    def values(variable):
        if "time" in variable.dims and variable.ndim > 1:
            reads.append(variable.shape)
        return original(variable)

    monkeypatch.setattr(xr.Variable, "values", property(values))
    assert series.get_entry("sample", 0.5, variables=()) == {}
    assert series.get_entry("sample", -1.0, variables=()) is None
    assert reads == []
    selected = series.get_entry("sample", 0.5, interpolation=interpolation, variables=("wind",))
    assert set(selected) == {"wind"}
    np.testing.assert_array_equal(
        selected["wind"], np.full(spaces["wind"].shape, 0.5 if interpolation else 0.0)
    )
    assert reads == [(2 if interpolation else 1, spaces["wind"].size)]
    reads.clear()
    queries = [0.5, 0.5, 0.0, 1.0, 1.0]
    block = series.get_entry("sample", queries, interpolation, variables=("wind",))["wind"]
    assert block.shape == spaces["wind"].shape + (len(queries),)
    assert reads == [(2, spaces["wind"].size)]
    with pytest.raises(KeyError, match="Unknown variables"):
        series.get_entry("sample", 0.5, variables=("missing",))


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
    filled = series.get_entry("sample", [0.0, 1.0], fill_value=0.0)["value"]
    np.testing.assert_array_equal(filled, np.zeros(space.shape + (2,)))


def test_timeseries_requires_field_space_and_name_only_variables():
    """Time-series schema keeps field types in CoefficientSpace only."""
    basis = GlobalCSBasis(4)
    field_space = CoefficientSpace(basis, representation="scalar")

    with pytest.raises(TypeError, match="field types belong in CoefficientSpace"):
        FieldTimeSeries({"sample": field_space}, {"sample": {"value": "scalar"}})

    with pytest.raises(ValueError, match="same keys"):
        FieldTimeSeries({"sample": field_space}, {"other": ("value",)})


def test_physical_microsecond_samples_are_distinct_and_causal():
    """Do not mistake a short physical interval for roundoff."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    for time, value in [(0, 0), (1, 1), (1 + 0.5e-6, 2)]:
        series.add_entry("sample", {"value": np.full(space.shape, value)}, time)
    assert series.datasets["sample"].sizes["time"] == 3
    for time, expected in [(1 - 0.5e-6, 0), (1, 1), (1 + 0.25e-6, 1), (1 + 0.5e-6, 2)]:
        np.testing.assert_array_equal(series.get_entry("sample", time)["value"], expected)


def test_intervals_merge_streams_but_skip_identical_records():
    """Changes combine independent clocks without spurious events."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries(
        {key: space for key in ("a", "b", "late")}, {key: ("value",) for key in ("a", "b", "late")}
    )
    records = {
        "a": [(0, 1), (0.15, 1), (0.3, 2), (0.4, 2)],
        "b": [(0, 0), (0.2, 1), (0.3, 2)],
        "late": [(0.25, 0)],
    }
    for key, entries in records.items():
        for time, value in entries:
            series.add_entry(key, {"value": np.full(space.shape, value)}, time)
    intervals = list(series.iter_intervals(0, 0.5))
    assert [(left, right) for left, right, _ in intervals] == [
        (0, 0.2),
        (0.2, 0.25),
        (0.25, 0.3),
        (0.3, 0.5),
        (0.5, 0.5),
    ]
    for left, _, entries in intervals:
        for key, values in entries.items():
            expected = series.get_entry(key, left)
            if expected is None:
                assert values is None
            else:
                np.testing.assert_array_equal(values["value"], expected["value"])
    assert intervals[0][2]["a"] is intervals[2][2]["a"]
    assert intervals[1][2]["b"] is intervals[2][2]["b"]
    assert [(a, b) for a, b, _ in series.iter_intervals(0.2, 0.3)] == [
        (0.2, 0.25),
        (0.25, 0.3),
        (0.3, 0.3),
    ]
    assert [(a, b) for a, b, _ in series.iter_intervals(3 * 0.1, 0.5)] == [
        (3 * 0.1, 0.5),
        (0.5, 0.5),
    ]
    np.testing.assert_array_equal(series.get_entry("a", 3 * 0.1)["value"], 2)


def test_intervals_read_each_coefficient_row_once_on_demand(monkeypatch):
    """Discover an event without materializing a coefficient history."""
    space = CoefficientSpace(SHBasis(2, 1))
    series = FieldTimeSeries({"sample": space}, {"sample": ("value",)})
    series.add_entries(
        "sample",
        {"value": np.broadcast_to(np.arange(100.0)[:, None], (100, space.size))},
        np.arange(100.0),
    )
    original = xr.Variable.values.fget
    rows = []

    def values(variable):
        if "i" in variable.dims:
            assert "time" not in variable.dims
            rows.append(original(variable).copy())
        return original(variable)

    monkeypatch.setattr(xr.Variable, "values", property(values))
    intervals = series.iter_intervals(0, 100)
    assert rows == []
    first = next(intervals)
    assert first[:2] == (0, 1)
    assert len(rows) == 2
    assert next(intervals)[:2] == (1, 2)
    assert len(rows) == 3
    list(intervals)
    assert len(rows) == 100
    np.testing.assert_array_equal(first[2]["sample"]["value"], 0)


@pytest.mark.parametrize("jump", [0.3, 3 * 0.1])
def test_intervals_include_endpoint_changes_and_preserve_distinct_nearby_times(jump):
    """The final snapshot is right-continuous, not a forcing step."""
    space = CoefficientSpace(SHBasis(1, 0))
    series = FieldTimeSeries(
        {key: space for key in ("a", "b")}, {key: ("v",) for key in ("a", "b")}
    )
    for key, onset in (("a", jump), ("b", 0.3)):
        for time, value in [(0, 1), (onset, 2), (onset + 1e-6, 3)]:
            series.add_entry(key, {"v": np.full(space.shape, value)}, time)
    intervals = list(series.iter_intervals(0, 0.3))
    assert [entry[:2] for entry in intervals] == [(0, 0.3), (0.3, 0.3)]
    for values in intervals[-1][2].values():
        np.testing.assert_array_equal(values["v"], 2)
    later = list(series.iter_intervals(0.3, 0.4))
    assert len(later) == 3
    np.testing.assert_allclose(later[1][0], 0.3 + 1e-6, rtol=0, atol=1e-15)


def test_interval_continuations_reuse_snapshots_but_see_edits_and_removed_streams():
    """Reuse is local to a selection, never a stale dataset cache."""
    space = CoefficientSpace(SHBasis(1, 0))
    series = FieldTimeSeries({"a": space}, {"a": ("v",)})
    series.add_entry("a", {"v": np.ones(space.shape)}, 1)
    previous = next(series.iter_intervals(1, 1))[2]
    same = next(series.iter_intervals(2, 2, previous=previous))[2]
    assert same["a"] is previous["a"]
    early = next(series.iter_intervals(0, 0, previous=previous))[2]
    assert early["a"] is None
    series.datasets["a"]["SH_v"].values *= 2
    changed = next(series.iter_intervals(2, 2, previous=previous))[2]
    np.testing.assert_array_equal(previous["a"]["v"], 1)
    np.testing.assert_array_equal(changed["a"]["v"], 2)
    series.datasets.clear()
    assert next(series.iter_intervals(2, 2, previous=changed))[2]["a"] is None


def test_intervals_allow_empty_series_and_reject_reversed_bounds():
    """Empty data still define a forward interval and endpoint."""
    series = FieldTimeSeries({}, {})
    assert list(series.iter_intervals(0, 1)) == [(0, 1, {}), (1, 1, {})]
    assert list(series.iter_intervals(1, 1)) == [(1, 1, {})]
    assert list(series.iter_intervals(3 * 0.1, 0.3)) == [(3 * 0.1, 3 * 0.1, {})]
    with pytest.raises(ValueError, match="stop"):
        list(series.iter_intervals(2, 1))
