
import pytest
from pynamit.simulation.runner import run_pynamit

def test_run_pynamit_dynamic_integration(tmp_path):
    """Verify run_pynamit accepts and applies dynamics_mode='full_induction'."""
    
    sim = run_pynamit(
        filename_prefix=str(tmp_path / "runner_integration"),
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
