"""Zarr persistence and restart behavior tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pynamit.sphere import SHBasis
from pynamit.default_run import run_pynamit
from pynamit.primitives.io import IO
from pynamit.primitives.field_space import FieldSpace
from pynamit.primitives.timeseries import Timeseries
from pynamit.simulation.dynamics import Dynamics


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


def _build_state_timeseries() -> Timeseries:
    sh_basis = SHBasis(2, 1)
    return Timeseries(
        {"state": FieldSpace(sh_basis, field_type="scalar")},
        {"state": ("m_ind", "m_imp")},
    )


def _add_state(ts: Timeseries, time: float, scale: float) -> None:
    n_coeffs = ts.get_field_space("state").index_length
    values = np.arange(n_coeffs, dtype=float) + scale
    ts.add_entry(
        "state",
        {"m_ind": values, "m_imp": -values},
        time,
    )


def _state_coefficients(dynamics: Dynamics) -> np.ndarray:
    state = dynamics.output_timeseries.datasets["state"]
    return np.hstack((state["SH_m_ind"].values[-1], state["SH_m_imp"].values[-1]))


def test_io_auto_uses_netcdf_when_zarr_is_unavailable(tmp_path, monkeypatch):
    """Auto storage remains usable without optional zarr installed."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: False))
    io = IO(tmp_path / "run")

    io.save_dataset(_small_dataset(), "state")

    assert io.get_dataset_storage_kind("state") == "netcdf"
    xr.testing.assert_equal(io.load_dataset("state"), _small_dataset())


def test_io_explicit_zarr_requires_dependency(tmp_path, monkeypatch):
    """Explicit zarr requests fail clearly when zarr is unavailable."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: False))
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")

    with pytest.raises(ImportError, match="optional 'zarr' dependency"):
        io.save_dataset(_small_dataset(), "state")


def test_io_auto_zarr_permission_error_is_not_silent(tmp_path, monkeypatch):
    """Permission failures should stop instead of switching format."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: True))

    def raising_to_zarr(self, store, *args, **kwargs):
        raise PermissionError("simulated zarr permission failure")

    monkeypatch.setattr(xr.Dataset, "to_zarr", raising_to_zarr)
    io = IO(tmp_path / "run")

    with pytest.raises(PermissionError, match="simulated zarr permission failure"):
        io.save_dataset(_small_dataset(), "state")

    assert not (tmp_path / "run" / "state.ncdf").exists()
    assert not (tmp_path / "run" / "state.zarr").exists()


def test_io_roundtrips_real_zarr_when_available(tmp_path):
    """Datasets and data arrays can be persisted as real zarr stores."""
    pytest.importorskip("zarr")
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")
    dataset = _small_dataset()
    dataarray = xr.DataArray(np.array([1.0, 2.0]), dims=["x"], name="PFAC_matrix")

    io.save_dataset(dataset, "state")
    io.save_dataarray(dataarray, "PFAC_matrix")

    assert io.get_dataset_storage_kind("state") == "zarr"
    assert (tmp_path / "run" / "state.zarr").is_dir()
    xr.testing.assert_equal(io.load_dataset("state"), dataset)
    xr.testing.assert_equal(io.load_dataarray("PFAC_matrix"), dataarray)


def test_io_zarr_writes_empty_chunks_for_strict_reads(tmp_path):
    """All-zero chunks are written for strict reads."""
    pytest.importorskip("zarr")
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")
    dataset = _small_dataset(np.zeros((1, 2), dtype=float))

    io.save_dataset(dataset, "state")

    xr.testing.assert_equal(io.load_dataset("state"), dataset)
    assert _first_data_chunk(tmp_path / "run" / "state.zarr", "value").exists()


def test_io_zarr_missing_chunk_raises_instead_of_filling(tmp_path):
    """Missing zarr chunks should fail loudly."""
    pytest.importorskip("zarr")
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")
    io.save_dataset(_small_dataset(), "state")
    _first_data_chunk(tmp_path / "run" / "state.zarr", "value").unlink()

    loaded = io.load_dataset("state")
    with pytest.raises(Exception) as excinfo:
        loaded["value"].values

    message = str(excinfo.value).lower()
    assert "chunk" in message or excinfo.type.__name__ in {"ChunkNotFoundError", "KeyError"}


def test_timeseries_save_appends_only_new_zarr_slices(tmp_path):
    """Loaded zarr time series append new monotonic samples."""
    pytest.importorskip("zarr")
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")
    ts = _build_state_timeseries()
    calls = []
    original_save_dataset = io.save_dataset

    def recording_save_dataset(dataset, name, print_info=False, *, storage=None, append_dim=None):
        calls.append((name, append_dim, int(dataset.sizes["time"])))
        return original_save_dataset(
            dataset, name, print_info=print_info, storage=storage, append_dim=append_dim
        )

    io.save_dataset = recording_save_dataset

    _add_state(ts, 0.0, 0.0)
    ts.save("state", io)
    _add_state(ts, 1.0, 10.0)
    ts.save("state", io)

    assert calls == [("state", None, 1), ("state", "time", 1)]
    loaded = io.load_dataset("state")
    np.testing.assert_allclose(loaded.time.values, [0.0, 1.0])


def test_timeseries_rewrites_zarr_for_same_time_replacement(tmp_path):
    """Replacing an existing timestamp requires a full store rewrite."""
    pytest.importorskip("zarr")
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")
    ts = _build_state_timeseries()
    calls = []
    original_save_dataset = io.save_dataset

    def recording_save_dataset(dataset, name, print_info=False, *, storage=None, append_dim=None):
        calls.append((name, append_dim, int(dataset.sizes["time"])))
        return original_save_dataset(
            dataset, name, print_info=print_info, storage=storage, append_dim=append_dim
        )

    io.save_dataset = recording_save_dataset

    _add_state(ts, 0.0, 0.0)
    ts.save("state", io)
    _add_state(ts, 0.0, 10.0)
    ts.save("state", io)

    assert calls == [("state", None, 1), ("state", None, 1)]
    loaded = io.load_dataset("state")
    n_coeffs = ts.get_field_space("state").index_length
    np.testing.assert_allclose(
        loaded["SH_m_ind"].values[0], np.arange(n_coeffs, dtype=float) + 10.0
    )


@pytest.mark.parametrize(
    ("backend", "data_source", "least_squares_solver"),
    [("numpy", "fallback", "normal_pinv")],
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
        mainfield_kind="dipole",
        ignore_PFAC=True,
        steady_state_initialization=False,
        artifact_storage="netcdf",
    )
    second = run_pynamit(
        final_time=0.0,
        dt=0.1,
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=True,
        steady_state_initialization=False,
        artifact_storage="netcdf",
    )

    assert first.run_directory != second.run_directory
    assert Path(first.run_directory, "settings.ncdf").is_file()
    assert Path(second.run_directory, "settings.ncdf").is_file()


@pytest.mark.parametrize(
    ("backend", "data_source", "least_squares_solver"),
    [("numpy", "fallback", "normal_pinv")],
)
@pytest.mark.parametrize("artifact_storage", ["netcdf", "zarr"])
def test_dynamics_restart_continues_to_match_direct_run(
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
        mainfield_kind="dipole",
        ignore_PFAC=True,
        use_wind=False,
        steady_state_initialization=False,
        plotsteps=1,
        artifact_storage=artifact_storage,
    )
    direct = run_pynamit(
        final_time=0.1,
        run_directory=str(tmp_path / f"direct-{artifact_storage}"),
        **common_kwargs,
    )
    partial_run_directory = tmp_path / f"restart-{artifact_storage}"
    run_pynamit(final_time=0.05, run_directory=str(partial_run_directory), **common_kwargs)

    resumed = Dynamics(
        run_directory=str(partial_run_directory),
        Nmax=4,
        Mmax=3,
        Ncs=8,
        mainfield_kind="dipole",
        ignore_PFAC=True,
        artifact_storage="auto",
    )
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
        resumed.output_timeseries.datasets["state"].time.values,
        direct.output_timeseries.datasets["state"].time.values,
    )
