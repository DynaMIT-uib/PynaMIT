
import pytest
from pathlib import Path
from pynamit.simulation.runner import run_pynamit

def test_run_pynamit_dynamic_integration(tmp_path):
    """Verify run_pynamit accepts and applies dynamics_mode='full_induction'."""
    
    sim = run_pynamit(
        run_directory=str(tmp_path / "runner_integration"),
        final_time=0.1, # Short run
        plotsteps=1,
        dt=0.1,
        Nmax=5,
        Mmax=2,
        dynamics_mode="full_induction",
        simulation_mode="pure_spectral", # Correct way to set mode
        mainfield_epoch=2020,
        mainfield_kind="igrf",
    )
    
    # Check if mode was set correctly
    assert sim.settings.dynamics_mode == "full_induction"
    
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
        mainfield_kind="igrf",
    )

    run_dir = Path(sim.run_directory)
    assert run_dir.parent == (tmp_path / "simulation")
    assert run_dir.name.startswith("run-")
    assert sim.uses_temporary_run_directory is False
    assert (run_dir / "settings.ncdf").exists()


def test_run_pynamit_accepts_run_directory(tmp_path):
    run_dir = tmp_path / "my_run"
    sim = run_pynamit(
        run_directory=run_dir,
        final_time=0.1,
        plotsteps=1,
        dt=0.1,
        Nmax=4,
        Mmax=2,
        mainfield_kind="igrf",
    )

    assert Path(sim.run_directory) == run_dir
    assert (run_dir / "settings.ncdf").exists()
