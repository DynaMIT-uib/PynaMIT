from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.data import SimulationData
from pynamit.simulation.settings import DynamicsSettings


def _artifact_path(run_dir: Path, name: str, storage_kind: str) -> Path:
    suffix = ".zarr" if storage_kind == "zarr" else ".ncdf"
    return run_dir / f"{name}{suffix}"


def _build_simulation_data(tmp_path: Path) -> SimulationData:
    run_dir = tmp_path / "timeseries_persistence"
    settings = DynamicsSettings(
        run_directory=str(run_dir),
        artifact_storage="zarr",
        Nmax=2,
        Mmax=2,
        Ncs=6,
        t0="2001-05-12 21:45:00",
    )
    return SimulationData.create(run_dir, settings, load_existing=False)


def _state_payload(n_solution: int, base: float) -> dict[str, np.ndarray]:
    return {
        "m_ind": np.full(n_solution, base + 1.0),
        "m_imp": np.full(n_solution, base + 2.0),
        "Phi": np.full(n_solution, base + 3.0),
        "W": np.full(n_solution, base + 4.0),
    }


class RecordingDatasetIO:
    def __init__(self, *, default_storage: str = "zarr") -> None:
        self.default_storage = default_storage
        self.storage_kinds: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def get_dataset_storage_kind(self, name: str) -> str | None:
        return self.storage_kinds.get(name)

    def default_dataset_storage_kind(self, name: str) -> str:
        return self.default_storage

    def save_dataset(
        self, dataset, name, print_info=False, *, storage=None, append_dim=None
    ) -> None:
        storage_kind = self.default_storage if storage is None else str(storage)
        self.calls.append(
            {
                "name": name,
                "storage": storage_kind,
                "append_dim": append_dim,
                "time_size": int(dataset.sizes.get("time", 0)),
            }
        )
        self.storage_kinds[name] = storage_kind


def test_timeseries_save_appends_only_new_zarr_slices(tmp_path):
    simulation_data = _build_simulation_data(tmp_path)
    timeseries = simulation_data.output_timeseries
    n_solution = simulation_data.solution_spec.index_length
    io = RecordingDatasetIO(default_storage="zarr")

    timeseries.add_entry("state", _state_payload(n_solution, 0.0), time=0.0)
    timeseries.save("state", io)

    timeseries.add_entry("state", _state_payload(n_solution, 10.0), time=1.0)
    timeseries.save("state", io)

    assert io.calls[0] == {"name": "state", "storage": "zarr", "append_dim": None, "time_size": 1}
    assert io.calls[1] == {
        "name": "state",
        "storage": "zarr",
        "append_dim": "time",
        "time_size": 1,
    }


def test_timeseries_rewrites_full_store_for_same_time_replacement(tmp_path):
    simulation_data = _build_simulation_data(tmp_path)
    timeseries = simulation_data.output_timeseries
    n_solution = simulation_data.solution_spec.index_length
    io = RecordingDatasetIO(default_storage="zarr")

    timeseries.add_entry("state", _state_payload(n_solution, 0.0), time=0.0)
    timeseries.save("state", io)

    timeseries.add_entry("state", _state_payload(n_solution, 20.0), time=0.0)
    timeseries.save("state", io)

    assert io.calls[1] == {"name": "state", "storage": "zarr", "append_dim": None, "time_size": 1}


def test_timeseries_use_real_zarr_store_when_available(tmp_path):
    pytest.importorskip("zarr")

    run_dir = tmp_path / "zarr_run"
    settings = DynamicsSettings(
        run_directory=str(run_dir),
        artifact_storage="zarr",
        Nmax=2,
        Mmax=2,
        Ncs=6,
        t0="2001-05-12 21:45:00",
    )
    simulation_data = SimulationData.create(run_dir, settings, load_existing=False)
    n_solution = simulation_data.solution_spec.index_length

    simulation_data.save_settings()
    simulation_data.save_pfac_matrix(np.eye(n_solution))
    simulation_data.add_output_entry("state", _state_payload(n_solution, 0.0), time=0.0)
    simulation_data.save_output_dataset("state")
    simulation_data.add_output_entry("state", _state_payload(n_solution, 10.0), time=1.0)
    simulation_data.save_output_dataset("state")

    assert simulation_data.io.get_dataset_storage_kind("settings") == "zarr"
    assert simulation_data.io.get_dataset_storage_kind("PFAC_matrix") == "zarr"
    assert simulation_data.io.get_dataset_storage_kind("state") == "zarr"
    assert _artifact_path(run_dir, "settings", "zarr").exists()
    assert _artifact_path(run_dir, "PFAC_matrix", "zarr").exists()
    assert _artifact_path(run_dir, "state", "zarr").exists()
    assert not _artifact_path(run_dir, "state", "netcdf").exists()

    reloaded = SimulationData.create(run_dir, settings, load_existing=True)
    assert reloaded.get_latest_output_time("state") == 1.0
    assert reloaded.io.get_dataset_storage_kind("settings") == "zarr"
    assert reloaded.io.get_dataset_storage_kind("PFAC_matrix") == "zarr"
    state_entry = reloaded.get_output_entry("state", 1.0)
    assert state_entry is not None
    np.testing.assert_allclose(state_entry["m_imp"], np.full(n_solution, 12.0))


def test_dynamics_restart_reads_state_from_real_zarr_store(tmp_path):
    pytest.importorskip("zarr")

    run_dir = tmp_path / "dynamics_restart_zarr"
    settings = DynamicsSettings(
        run_directory=str(run_dir), Nmax=2, Mmax=2, Ncs=6, t0="2001-05-12 21:45:00"
    )
    dynamics = Dynamics(settings, benchmark_mode=False)
    n_solution = dynamics.data.solution_spec.index_length

    dynamics.data.add_output_entry("state", _state_payload(n_solution, 0.0), time=0.0)
    dynamics.data.save_output_dataset("state")
    expected_state = _state_payload(n_solution, 10.0)
    dynamics.data.add_output_entry("state", expected_state, time=2.0)
    dynamics.data.save_output_dataset("state")

    assert _artifact_path(run_dir, "settings", "zarr").exists()
    assert _artifact_path(run_dir, "PFAC_matrix", "zarr").exists()
    assert _artifact_path(run_dir, "state", "zarr").exists()
    assert not _artifact_path(run_dir, "state", "netcdf").exists()

    resumed = Dynamics.from_directory(run_dir)

    assert resumed.io.get_dataset_storage_kind("settings") == "zarr"
    assert resumed.io.get_dataset_storage_kind("PFAC_matrix") == "zarr"
    assert resumed.io.get_dataset_storage_kind("state") == "zarr"
    assert float(resumed.current_time) == 2.0
    np.testing.assert_allclose(np.asarray(resumed._current_m_ind), expected_state["m_ind"])
    state_entry = resumed.data.get_output_entry("state", resumed.current_time)
    assert state_entry is not None
    np.testing.assert_allclose(state_entry["m_imp"], expected_state["m_imp"])
