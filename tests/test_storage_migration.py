from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pynamit.primitives.io import IO
from pynamit.simulation.data import SimulationData
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.migration import migrate_run_storage
from pynamit.simulation.settings import DynamicsSettings

pytest.importorskip("zarr")


def _artifact_path(run_dir: Path, name: str, storage_kind: str) -> Path:
    suffix = ".zarr" if storage_kind == "zarr" else ".ncdf"
    return run_dir / f"{name}{suffix}"


def _state_payload(n_solution: int, base: float) -> dict[str, np.ndarray]:
    return {
        "m_ind": np.full(n_solution, base + 1.0),
        "m_imp": np.full(n_solution, base + 2.0),
        "Phi": np.full(n_solution, base + 3.0),
        "W": np.full(n_solution, base + 4.0),
    }


def _create_run_with_storage(
    run_dir: Path, *, settings_storage: str, pfac_storage: str, jr_storage: str, state_storage: str
) -> tuple[SimulationData, int]:
    settings = DynamicsSettings(
        run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6, t0="2001-05-12 21:45:00"
    )
    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    io = simulation_data.io

    n_solution = simulation_data.solution_spec.index_length
    n_scalar = simulation_data.sh_basis.scalar_index_length(mean_free=True)

    io.save_dataset(settings.to_dataset(), "settings", storage=settings_storage)
    io.save_dataarray(
        xr.DataArray(np.eye(n_solution), name="PFAC_matrix"), "PFAC_matrix", storage=pfac_storage
    )

    simulation_data.input_timeseries.add_entry(
        "jr", {"jr": np.arange(n_scalar, dtype=float)}, time=0.0
    )
    io.save_dataset(
        simulation_data.input_timeseries.datasets["jr"].reset_index("i"), "jr", storage=jr_storage
    )

    simulation_data.add_output_entry("state", _state_payload(n_solution, 0.0), time=0.0)
    expected_state = _state_payload(n_solution, 10.0)
    simulation_data.add_output_entry("state", expected_state, time=2.0)
    io.save_dataset(
        simulation_data.output_timeseries.datasets["state"].reset_index("i"),
        "state",
        storage=state_storage,
    )

    return simulation_data, n_solution


def test_migrate_run_storage_netcdf_to_zarr(tmp_path):
    run_dir = tmp_path / "netcdf_to_zarr"
    _, n_solution = _create_run_with_storage(
        run_dir,
        settings_storage="netcdf",
        pfac_storage="netcdf",
        jr_storage="netcdf",
        state_storage="netcdf",
    )

    report = migrate_run_storage(run_dir, "zarr")

    assert set(report.migrated_artifacts) == {"PFAC_matrix", "jr", "settings", "state"}
    assert report.unchanged_artifacts == ()
    for name in ("settings", "PFAC_matrix", "jr", "state"):
        assert _artifact_path(run_dir, name, "zarr").exists()
        assert not _artifact_path(run_dir, name, "netcdf").exists()

    resumed = Dynamics.from_directory(run_dir)
    assert resumed.io.get_dataset_storage_kind("settings") == "zarr"
    assert resumed.io.get_dataset_storage_kind("PFAC_matrix") == "zarr"
    assert resumed.io.get_dataset_storage_kind("jr") == "zarr"
    assert resumed.io.get_dataset_storage_kind("state") == "zarr"
    assert float(resumed.current_time) == 2.0
    np.testing.assert_allclose(np.asarray(resumed._current_m_ind), np.full(n_solution, 11.0))


def test_migrate_run_storage_zarr_to_netcdf(tmp_path):
    run_dir = tmp_path / "zarr_to_netcdf"
    _, n_solution = _create_run_with_storage(
        run_dir,
        settings_storage="zarr",
        pfac_storage="zarr",
        jr_storage="zarr",
        state_storage="zarr",
    )

    report = migrate_run_storage(run_dir, "netcdf")

    assert set(report.migrated_artifacts) == {"PFAC_matrix", "jr", "settings", "state"}
    assert report.unchanged_artifacts == ()
    for name in ("settings", "PFAC_matrix", "jr", "state"):
        assert _artifact_path(run_dir, name, "netcdf").exists()
        assert not _artifact_path(run_dir, name, "zarr").exists()

    resumed = Dynamics.from_directory(run_dir)
    assert resumed.io.get_dataset_storage_kind("settings") == "netcdf"
    assert resumed.io.get_dataset_storage_kind("PFAC_matrix") == "netcdf"
    assert resumed.io.get_dataset_storage_kind("jr") == "netcdf"
    assert resumed.io.get_dataset_storage_kind("state") == "netcdf"
    assert float(resumed.current_time) == 2.0
    np.testing.assert_allclose(np.asarray(resumed._current_m_ind), np.full(n_solution, 11.0))


def test_migrate_run_storage_converts_only_non_target_artifacts(tmp_path):
    run_dir = tmp_path / "mixed_to_zarr"
    _create_run_with_storage(
        run_dir,
        settings_storage="netcdf",
        pfac_storage="zarr",
        jr_storage="zarr",
        state_storage="netcdf",
    )

    report = migrate_run_storage(run_dir, "zarr")

    assert set(report.migrated_artifacts) == {"settings", "state"}
    assert set(report.unchanged_artifacts) == {"PFAC_matrix", "jr"}
    for name in ("settings", "PFAC_matrix", "jr", "state"):
        assert _artifact_path(run_dir, name, "zarr").exists()
        assert not _artifact_path(run_dir, name, "netcdf").exists()


def test_migrate_run_storage_rejects_dual_format_artifacts(tmp_path):
    run_dir = tmp_path / "dual_format"
    simulation_data, _ = _create_run_with_storage(
        run_dir,
        settings_storage="netcdf",
        pfac_storage="netcdf",
        jr_storage="netcdf",
        state_storage="netcdf",
    )

    simulation_data.io.save_dataset(
        simulation_data.settings.to_dataset(), "settings", storage="zarr"
    )

    with pytest.raises(ValueError, match="exists as both NetCDF and Zarr"):
        migrate_run_storage(run_dir, "zarr")


def test_migrate_run_storage_ignores_unknown_top_level_artifacts(tmp_path):
    run_dir = tmp_path / "ignore_unknown"
    _create_run_with_storage(
        run_dir,
        settings_storage="netcdf",
        pfac_storage="netcdf",
        jr_storage="netcdf",
        state_storage="netcdf",
    )
    unknown_path = run_dir / "notes.ncdf"
    unknown_path.write_text("not a pynamit artifact", encoding="utf-8")

    report = migrate_run_storage(run_dir, "zarr")

    assert set(report.migrated_artifacts) == {"PFAC_matrix", "jr", "settings", "state"}
    assert unknown_path.exists()
