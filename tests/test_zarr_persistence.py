"""Persistence tests for NetCDF/Zarr storage selection and restarts."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.primitives.io import IO
from pynamit.primitives.timeseries import Timeseries
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.migration import migrate_file_prefix_storage
from pynamit.spherical_harmonics.sh_basis import SHBasis


def test_io_auto_uses_netcdf_when_zarr_is_unavailable(tmp_path, monkeypatch):
    """Auto storage should work without optional zarr."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: False))
    prefix = tmp_path / "run"
    io = IO(prefix)
    dataset = xr.Dataset({"value": ("x", np.array([1.0, 2.0]))}, coords={"x": [0, 1]})

    io.save_dataset(dataset, "state")

    assert (tmp_path / "run_state.ncdf").is_file()
    assert not (tmp_path / "run_state.zarr").exists()
    assert io.get_dataset_storage_kind("state") == "netcdf"
    xr.testing.assert_equal(io.load_dataset("state"), dataset)


def test_io_explicit_zarr_requires_dependency(tmp_path, monkeypatch):
    """Explicit zarr writes should fail clearly."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: False))
    io = IO(tmp_path / "run")
    dataset = xr.Dataset({"value": ("x", np.array([1.0]))})

    with pytest.raises(ImportError, match="optional 'zarr' dependency"):
        io.save_dataset(dataset, "state", storage="zarr")


def test_io_existing_zarr_requires_dependency_for_auto_load(tmp_path, monkeypatch):
    """Auto loads should fail clearly for zarr without zarr."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: False))
    (tmp_path / "run_state.zarr").mkdir()
    io = IO(tmp_path / "run")

    with pytest.raises(ImportError, match="optional 'zarr' dependency"):
        io.load_dataset("state")


def test_io_auto_falls_back_to_netcdf_on_zarr_permission_error(tmp_path, monkeypatch):
    """Auto storage can recover from zarr write denial."""
    monkeypatch.setattr(IO, "zarr_available", staticmethod(lambda: True))
    io = IO(tmp_path / "run")
    dataset = xr.Dataset({"value": ("x", np.array([1.0]))})

    def raising_to_zarr(self, store, *args, **kwargs):
        raise PermissionError("simulated zarr permission failure")

    monkeypatch.setattr(xr.Dataset, "to_zarr", raising_to_zarr)

    with pytest.warns(RuntimeWarning, match="Falling back to NetCDF"):
        io.save_dataset(dataset, "state")

    assert io.get_dataset_storage_kind("state") == "netcdf"
    assert (tmp_path / "run_state.ncdf").is_file()
    assert not (tmp_path / "run_state.zarr").exists()


class RecordingDatasetIO:
    """Minimal IO double that records timeseries save calls."""

    def __init__(self, *, default_storage: str = "zarr") -> None:
        self.default_storage = default_storage
        self.storage_by_name: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def get_dataset_storage_kind(self, name: str) -> str | None:
        """Return the stored kind for one artifact."""
        return self.storage_by_name.get(name)

    def default_dataset_storage_kind(self, name: str) -> str:
        """Return the default storage kind."""
        return self.default_storage

    def save_dataset(
        self,
        dataset: xr.Dataset,
        name: str,
        print_info: bool = False,
        *,
        storage: str | None = None,
        append_dim: str | None = None,
    ) -> None:
        """Record one dataset save call."""
        storage_kind = self.default_storage if storage is None else storage
        self.calls.append(
            {
                "name": name,
                "storage": storage_kind,
                "append_dim": append_dim,
                "time_size": int(dataset.sizes.get("time", 0)),
            }
        )
        self.storage_by_name[name] = storage_kind


def _build_state_timeseries() -> tuple[Timeseries, int]:
    cs_basis = CSBasis(4)
    sh_basis = SHBasis(2, 1)
    timeseries = Timeseries(cs_basis, {"state": sh_basis}, {"state": {"m_ind": "scalar"}})
    return timeseries, sh_basis.index_length


def test_timeseries_save_appends_only_new_zarr_slices():
    """Chronological zarr saves should append new slices."""
    timeseries, n_coefficients = _build_state_timeseries()
    io = RecordingDatasetIO(default_storage="zarr")

    timeseries.add_entry("state", {"m_ind": np.zeros(n_coefficients)}, time=0.0)
    timeseries.save("state", io)
    timeseries.add_entry("state", {"m_ind": np.ones(n_coefficients)}, time=1.0)
    timeseries.save("state", io)

    assert io.calls == [
        {"name": "state", "storage": "zarr", "append_dim": None, "time_size": 1},
        {"name": "state", "storage": "zarr", "append_dim": "time", "time_size": 1},
    ]


def test_timeseries_rewrites_full_store_for_same_time_replacement():
    """Replacing an existing slice should rewrite the store."""
    timeseries, n_coefficients = _build_state_timeseries()
    io = RecordingDatasetIO(default_storage="zarr")

    timeseries.add_entry("state", {"m_ind": np.zeros(n_coefficients)}, time=0.0)
    timeseries.save("state", io)
    timeseries.add_entry("state", {"m_ind": np.ones(n_coefficients)}, time=0.0)
    timeseries.save("state", io)

    assert io.calls[-1] == {"name": "state", "storage": "zarr", "append_dim": None, "time_size": 1}


def test_io_roundtrips_real_zarr_when_available(tmp_path):
    """Real zarr stores should round-trip through IO."""
    pytest.importorskip("zarr")
    io = IO(tmp_path / "run", preferred_dataset_storage="zarr")
    dataset = xr.Dataset(
        {"value": (("time", "x"), np.array([[1.0, 2.0]]))}, coords={"time": [0.0], "x": [0, 1]}
    )

    io.save_dataset(dataset, "state")

    assert (tmp_path / "run_state.zarr").is_dir()
    assert io.get_dataset_storage_kind("state") == "zarr"
    xr.testing.assert_equal(io.load_dataset("state"), dataset)


def test_migrate_file_prefix_storage_reports_unchanged_netcdf(tmp_path):
    """Migration reports artifacts already in the target format."""
    prefix = tmp_path / "run"
    io = IO(prefix, preferred_dataset_storage="netcdf")
    settings = xr.Dataset(attrs={"Nmax": 2})
    io.save_dataset(settings, "settings")

    report = migrate_file_prefix_storage(prefix, "netcdf")

    assert report.target_storage == "netcdf"
    assert report.migrated_artifacts == ()
    assert report.unchanged_artifacts == ("settings",)


def test_migrate_file_prefix_storage_netcdf_to_real_zarr(tmp_path):
    """Migration converts NetCDF artifacts to zarr when available."""
    pytest.importorskip("zarr")
    prefix = tmp_path / "run"
    io = IO(prefix, preferred_dataset_storage="netcdf")
    settings = xr.Dataset(attrs={"Nmax": 2})
    state = xr.Dataset({"value": ("time", np.array([1.0]))}, coords={"time": [0.0]})
    pfac = xr.DataArray(np.eye(2), dims=("i", "j"))
    io.save_dataset(settings, "settings")
    io.save_dataset(state, "state")
    io.save_dataarray(pfac, "PFAC_matrix")

    report = migrate_file_prefix_storage(prefix, "zarr")

    assert report.migrated_artifacts == ("PFAC_matrix", "settings", "state")
    assert not (tmp_path / "run_settings.ncdf").exists()
    assert (tmp_path / "run_settings.zarr").is_dir()
    xr.testing.assert_equal(io.load_dataset("state"), state)
    xr.testing.assert_equal(io.load_dataarray("PFAC_matrix"), pfac)


def _dynamics_kwargs(prefix, *, artifact_storage: str) -> dict[str, object]:
    return {
        "filename_prefix": str(prefix),
        "Nmax": 2,
        "Mmax": 1,
        "Ncs": 4,
        "mainfield_kind": "dipole",
        "ignore_PFAC": True,
        "connect_hemispheres": False,
        "save_steady_states": False,
        "artifact_storage": artifact_storage,
        "backend": "numpy",
    }


def test_dynamics_restart_reads_saved_state_from_existing_storage(tmp_path):
    """Restart should recover the latest saved state time."""
    prefix = tmp_path / "restart"
    dynamics = Dynamics(**_dynamics_kwargs(prefix, artifact_storage="netcdf"))
    n_coefficients = dynamics.output_storage_bases["state"].index_length
    expected_state = {
        "m_ind": np.arange(n_coefficients, dtype=float),
        "m_imp": np.arange(n_coefficients, dtype=float) + 10.0,
        "Phi": np.arange(n_coefficients, dtype=float) + 20.0,
        "W": np.arange(n_coefficients, dtype=float) + 30.0,
    }
    dynamics.output_timeseries.add_entry("state", expected_state, time=3.0)
    dynamics.output_timeseries.save("state", dynamics.io)

    resumed = Dynamics(**_dynamics_kwargs(prefix, artifact_storage="auto"))
    state_entry = resumed.output_timeseries.get_entry("state", 3.0)

    assert resumed.current_time == pytest.approx(3.0)
    assert resumed.io.get_dataset_storage_kind("state") == "netcdf"
    for key, expected in expected_state.items():
        np.testing.assert_allclose(state_entry[key], expected)


def test_dynamics_restart_reads_saved_state_from_real_zarr_store(tmp_path):
    """Dynamics restart should read state from zarr."""
    pytest.importorskip("zarr")
    prefix = tmp_path / "zarr_restart"
    dynamics = Dynamics(**_dynamics_kwargs(prefix, artifact_storage="zarr"))
    n_coefficients = dynamics.output_storage_bases["state"].index_length
    expected_state = {
        "m_ind": np.arange(n_coefficients, dtype=float),
        "m_imp": np.arange(n_coefficients, dtype=float) + 10.0,
        "Phi": np.arange(n_coefficients, dtype=float) + 20.0,
        "W": np.arange(n_coefficients, dtype=float) + 30.0,
    }
    dynamics.output_timeseries.add_entry("state", expected_state, time=4.0)
    dynamics.output_timeseries.save("state", dynamics.io)

    resumed = Dynamics(**_dynamics_kwargs(prefix, artifact_storage="auto"))
    state_entry = resumed.output_timeseries.get_entry("state", 4.0)

    assert resumed.current_time == pytest.approx(4.0)
    assert resumed.io.get_dataset_storage_kind("state") == "zarr"
    for key, expected in expected_state.items():
        np.testing.assert_allclose(state_entry[key], expected)
