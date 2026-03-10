import pytest
from pathlib import Path
from pynamit.primitives.io import IO
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, MainfieldKind, SimulationMode


def _settings_path(run_dir: Path, storage_kind: str) -> Path:
    suffix = ".zarr" if storage_kind == "zarr" else ".ncdf"
    return run_dir / f"settings{suffix}"


def test_run_pynamit_dynamic_integration(tmp_path):
    """Verify run_pynamit accepts and applies dynamics_mode=DynamicsMode.FULL_INDUCTION."""

    sim = run_pynamit(
        run_directory=str(tmp_path / "runner_integration"),
        final_time=0.1,  # Short run
        plotsteps=1,
        dt=0.1,
        Nmax=5,
        Mmax=2,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,  # Correct way to set mode
        mainfield_epoch=2020,
        mainfield_kind=MainfieldKind.IGRF,
    )

    # Check if mode was set correctly
    assert sim.settings.dynamics_mode == DynamicsMode.FULL_INDUCTION

    # Check if necessary state variables are present
    assert sim.state.psi is not None


def test_run_pynamit_uses_simulation_directory_for_default_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = run_pynamit(
        run_directory=None,
        final_time=0.1,
        plotsteps=1,
        dt=0.1,
        Nmax=4,
        Mmax=2,
        mainfield_kind=MainfieldKind.IGRF,
    )

    run_dir = Path(sim.run_directory)
    assert run_dir.parent == (tmp_path / "simulation")
    assert run_dir.name.startswith("run-")
    assert sim.uses_temporary_run_directory is False
    settings_storage = sim.io.get_dataset_storage_kind("settings")
    assert settings_storage == ("zarr" if IO.zarr_available() else "netcdf")
    assert _settings_path(run_dir, settings_storage).exists()


def test_run_pynamit_accepts_run_directory(tmp_path):
    run_dir = tmp_path / "my_run"
    sim = run_pynamit(
        run_directory=run_dir,
        final_time=0.1,
        plotsteps=1,
        dt=0.1,
        Nmax=4,
        Mmax=2,
        mainfield_kind=MainfieldKind.IGRF,
    )

    assert Path(sim.run_directory) == run_dir
    settings_storage = sim.io.get_dataset_storage_kind("settings")
    assert settings_storage == ("zarr" if IO.zarr_available() else "netcdf")
    assert _settings_path(run_dir, settings_storage).exists()
