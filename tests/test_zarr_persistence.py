"""Zarr persistence and restart behavior tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pynamit.fields import FieldSpace
from pynamit.simulation.api import Simulation
from pynamit.simulation.workflows.standard import run_pynamit
from pynamit.sphere import SHBasis
from pynamit.storage import ArtifactStore, FieldTimeSeries


def _small_dataset(values: np.ndarray | None = None) -> xr.Dataset:
    if values is None:
        values = np.array([[1.0, 2.0]])
    return xr.Dataset(
        {"value": (("time", "x"), values)},
        coords={"time": np.arange(values.shape[0], dtype=float), "x": np.arange(values.shape[1])},
    )


def _first_data_chunk(store: Path, variable_name: str) -> Path:
    metadata_names = {".zarray", ".zattrs", ".zgroup", "zarr.json"}
    chunks = [
        path
        for path in (store / variable_name).rglob("*")
        if path.is_file() and path.name not in metadata_names
    ]
    assert chunks, f"No chunk files found for {variable_name!r} in {store}"
    return chunks[0]


def _build_state_timeseries() -> FieldTimeSeries:
    sh_basis = SHBasis(2, 1)
    return FieldTimeSeries(
        {"state": FieldSpace(sh_basis, field_type="scalar")}, {"state": ("m_ind", "m_imp")}
    )


def _add_state(ts: FieldTimeSeries, time: float, scale: float) -> None:
    n_coeffs = ts.get_field_space("state").index_length
    values = np.arange(n_coeffs, dtype=float) + scale
    ts.add_entry("state", {"m_ind": values, "m_imp": -values}, time)


def _state_coefficients(simulation: Simulation) -> np.ndarray:
    state = simulation.run_data.output_series.datasets["state"]
    return np.hstack((state["SH_m_ind"].values[-1], state["SH_m_imp"].values[-1]))


def test_artifact_store_auto_uses_netcdf_when_zarr_is_unavailable(tmp_path, monkeypatch):
    """Auto storage remains usable without optional zarr installed."""
    monkeypatch.setattr(ArtifactStore, "zarr_available", staticmethod(lambda: False))
    store = ArtifactStore(tmp_path / "run")

    store.save_dataset(_small_dataset(), "state")

    assert store.get_dataset_storage_kind("state") == "netcdf"
    xr.testing.assert_equal(store.load_dataset("state"), _small_dataset())


def test_artifact_store_explicit_zarr_requires_dependency(tmp_path, monkeypatch):
    """Explicit zarr requests fail clearly when zarr is unavailable."""
    monkeypatch.setattr(ArtifactStore, "zarr_available", staticmethod(lambda: False))
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="zarr")

    with pytest.raises(ImportError, match="optional 'zarr' dependency"):
        store.save_dataset(_small_dataset(), "state")


def test_artifact_store_scans_only_explicit_artifact_names(tmp_path):
    """Generic persistence does not own simulation artifact names."""
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="netcdf")
    store.save_dataset(_small_dataset(), "settings")
    store.save_dataset(_small_dataset(), "state")

    assert store.scan_artifacts(("state",)) == {"state": ("netcdf",)}


def test_artifact_store_rejects_ambiguous_artifact_representations(tmp_path):
    """Reads reject duplicate physical artifacts."""
    store = ArtifactStore(tmp_path / "run")
    run_directory = Path(store.directory)
    run_directory.mkdir(parents=True)
    (run_directory / "state.ncdf").touch()
    (run_directory / "state.zarr").mkdir()

    with pytest.raises(ValueError, match="ambiguous storage representations"):
        store.get_dataset_storage_kind("state")


def test_artifact_store_requires_only_explicitly_named_artifacts(tmp_path):
    """Directory validation remains artifact-generic."""
    run_directory = tmp_path / "run"
    store = ArtifactStore(run_directory, preferred_dataset_storage="netcdf")
    store.save_dataset(_small_dataset(), "custom")

    assert ArtifactStore.require_artifact_directory(run_directory, ("custom",)) == str(
        run_directory.resolve()
    )
    with pytest.raises(ValueError, match=r"missing required artifact\(s\): \['settings'\]"):
        ArtifactStore.require_artifact_directory(run_directory, ("settings",))
    with pytest.raises(TypeError, match="collection"):
        ArtifactStore.require_artifact_directory(run_directory, "custom")


@pytest.mark.parametrize("name", ["", ".", "..", "../state", "nested/state", r"nested\state"])
def test_artifact_store_rejects_nonlocal_artifact_names(tmp_path, name):
    """Artifact names cannot escape or introduce directory structure."""
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="netcdf")

    with pytest.raises(ValueError, match="path-safe"):
        store.save_dataset(_small_dataset(), name)


def test_artifact_store_auto_zarr_permission_error_is_not_silent(tmp_path, monkeypatch):
    """Permission failures should stop instead of switching format."""
    monkeypatch.setattr(ArtifactStore, "zarr_available", staticmethod(lambda: True))

    def raising_to_zarr(self, store, *args, **kwargs):
        raise PermissionError("simulated zarr permission failure")

    monkeypatch.setattr(xr.Dataset, "to_zarr", raising_to_zarr)
    store = ArtifactStore(tmp_path / "run")

    with pytest.raises(PermissionError, match="simulated zarr permission failure"):
        store.save_dataset(_small_dataset(), "state")

    assert not (tmp_path / "run" / "state.ncdf").exists()
    assert not (tmp_path / "run" / "state.zarr").exists()


def test_artifact_store_netcdf_write_failure_cleans_unique_temporary_file(tmp_path, monkeypatch):
    """Failed NetCDF writes leave neither target nor temporary files."""
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="netcdf")

    def raising_to_netcdf(self, path, *args, **kwargs):
        raise OSError("simulated NetCDF write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", raising_to_netcdf)

    with pytest.raises(OSError, match="simulated NetCDF write failure"):
        store.save_dataset(_small_dataset(), "state")

    assert not (tmp_path / "run" / "state.ncdf").exists()
    assert list((tmp_path / "run").iterdir()) == []


def test_artifact_store_roundtrips_real_zarr_when_available(tmp_path):
    """Datasets and data arrays can be persisted as real zarr stores."""
    pytest.importorskip("zarr")
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="zarr")
    dataset = _small_dataset()
    dataarray = xr.DataArray(np.array([1.0, 2.0]), dims=["x"], name="PFAC_matrix")

    store.save_dataset(dataset, "state")
    store.save_dataarray(dataarray, "PFAC_matrix")

    assert store.get_dataset_storage_kind("state") == "zarr"
    assert (tmp_path / "run" / "state.zarr").is_dir()
    xr.testing.assert_equal(store.load_dataset("state"), dataset)
    xr.testing.assert_equal(store.load_dataarray("PFAC_matrix"), dataarray)


def test_artifact_store_format_change_removes_stale_artifact_representation(tmp_path):
    """One artifact retains only its current storage representation."""
    pytest.importorskip("zarr")
    store = ArtifactStore(tmp_path / "run")
    first = _small_dataset(np.array([[1.0, 2.0]]))
    second = _small_dataset(np.array([[3.0, 4.0]]))

    store.save_dataset(first, "state", storage="zarr")
    store.save_dataset(second, "state", storage="netcdf")

    assert store.get_dataset_storage_kinds("state") == ("netcdf",)
    xr.testing.assert_equal(store.load_dataset("state"), second)


def test_artifact_store_zarr_writes_empty_chunks_for_strict_reads(tmp_path):
    """All-zero chunks are written for strict reads."""
    pytest.importorskip("zarr")
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="zarr")
    dataset = _small_dataset(np.zeros((1, 2), dtype=float))

    store.save_dataset(dataset, "state")

    xr.testing.assert_equal(store.load_dataset("state"), dataset)
    assert _first_data_chunk(tmp_path / "run" / "state.zarr", "value").exists()


def test_artifact_store_zarr_missing_chunk_raises_instead_of_filling(tmp_path):
    """Missing zarr chunks should fail loudly."""
    pytest.importorskip("zarr")
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="zarr")
    store.save_dataset(_small_dataset(), "state")
    _first_data_chunk(tmp_path / "run" / "state.zarr", "value").unlink()

    loaded = store.load_dataset("state")
    with pytest.raises(Exception) as excinfo:
        _ = loaded["value"].values

    message = str(excinfo.value).lower()
    assert "chunk" in message or excinfo.type.__name__ in {"ChunkNotFoundError", "KeyError"}


def test_timeseries_save_appends_only_new_zarr_slices(tmp_path):
    """Loaded zarr time series append new monotonic samples."""
    pytest.importorskip("zarr")
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="zarr")
    ts = _build_state_timeseries()
    calls = []
    original_save_dataset = store.save_dataset

    def recording_save_dataset(dataset, name, print_info=False, *, storage=None, append_dim=None):
        calls.append((name, append_dim, int(dataset.sizes["time"])))
        return original_save_dataset(
            dataset, name, print_info=print_info, storage=storage, append_dim=append_dim
        )

    store.save_dataset = recording_save_dataset

    _add_state(ts, 0.0, 0.0)
    ts.save("state", store)
    _add_state(ts, 1.0, 10.0)
    ts.save("state", store)

    assert calls == [("state", None, 1), ("state", "time", 1)]
    loaded = store.load_dataset("state")
    np.testing.assert_allclose(loaded.time.values, [0.0, 1.0])


def test_timeseries_rewrites_zarr_for_same_time_replacement(tmp_path):
    """Replacing an existing timestamp requires a full store rewrite."""
    pytest.importorskip("zarr")
    store = ArtifactStore(tmp_path / "run", preferred_dataset_storage="zarr")
    ts = _build_state_timeseries()
    calls = []
    original_save_dataset = store.save_dataset

    def recording_save_dataset(dataset, name, print_info=False, *, storage=None, append_dim=None):
        calls.append((name, append_dim, int(dataset.sizes["time"])))
        return original_save_dataset(
            dataset, name, print_info=print_info, storage=storage, append_dim=append_dim
        )

    store.save_dataset = recording_save_dataset

    _add_state(ts, 0.0, 0.0)
    ts.save("state", store)
    _add_state(ts, 0.0, 10.0)
    ts.save("state", store)

    assert calls == [("state", None, 1), ("state", None, 1)]
    loaded = store.load_dataset("state")
    n_coeffs = ts.get_field_space("state").index_length
    np.testing.assert_allclose(
        loaded["SH_m_ind"].values[0], np.arange(n_coeffs, dtype=float) + 10.0
    )


@pytest.mark.parametrize(
    ("backend", "data_source", "least_squares_solver"), [("numpy", "fallback", "normal_pinv")]
)
def test_run_pynamit_default_run_directories_are_isolated(
    backend, data_source, least_squares_solver
):
    """Default runs should not reuse fixed artifact paths."""
    first = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        steady_state_initialization=False,
        artifact_storage="netcdf",
    )
    second = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        steady_state_initialization=False,
        artifact_storage="netcdf",
    )

    assert first.run_data.run_directory != second.run_data.run_directory
    assert Path(first.run_data.run_directory, "settings.ncdf").is_file()
    assert Path(second.run_data.run_directory, "settings.ncdf").is_file()


@pytest.mark.parametrize(
    ("backend", "data_source", "least_squares_solver"), [("numpy", "fallback", "normal_pinv")]
)
@pytest.mark.parametrize("artifact_storage", ["netcdf", "zarr"])
def test_simulation_restart_continues_to_match_direct_run(
    tmp_path, backend, data_source, least_squares_solver, artifact_storage
):
    """Restarting should match the direct continuation."""
    if artifact_storage == "zarr":
        pytest.importorskip("zarr")

    common_kwargs = dict(
        dt=0.05,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        main_field_kind="dipole",
        enable_pfac_coupling=False,
        use_wind=False,
        steady_state_initialization=False,
        saving_sample_interval=1,
        artifact_storage=artifact_storage,
    )
    direct = run_pynamit(
        final_time=0.1, run_directory=str(tmp_path / f"direct-{artifact_storage}"), **common_kwargs
    )
    partial_run_directory = tmp_path / f"restart-{artifact_storage}"
    run_pynamit(final_time=0.05, run_directory=str(partial_run_directory), **common_kwargs)

    resumed = Simulation.from_directory(str(partial_run_directory), artifact_storage="auto")
    resumed.evolve_to_time(
        t=0.1,
        dt=0.05,
        sampling_step_interval=1,
        saving_sample_interval=1,
        steady_state_initialization=False,
        quiet=True,
    )

    np.testing.assert_allclose(
        _state_coefficients(resumed), _state_coefficients(direct), rtol=1e-10, atol=0.0
    )
    np.testing.assert_allclose(
        resumed.run_data.output_series.datasets["state"].time.values,
        direct.run_data.output_series.datasets["state"].time.values,
    )
