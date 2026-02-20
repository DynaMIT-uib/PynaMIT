"""Tests for matrix-free iterative solver paths in full induction mode."""

import pytest
import numpy as np
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.dynamics import SimulationMode

@pytest.mark.parametrize("solver", ["lsmr", "cg"])
def test_full_induction_iterative_solvers(solver):
    """Verify that full induction runs successfully with iterative solvers.
    
    This ensures the matrix-free path (State.steady_state_coupled with LinearMap)
    is correctly exercised and converges, unlike test_dynamic_solver which forces SVD.
    """
    # Run a short simulation
    sim = run_pynamit(
        final_time=0.01,
        dt=0.01,
        Nmax=8,
        Mmax=4,
        Ncs=16,
        dynamics_mode="full_induction",
        # Use simple spectral mode for speed
        simulation_mode="pure_spectral",
        ignore_PFAC=False,
        mainfield_kind="igrf",
        connect_hemispheres=True,
        # Explicitly request iterative solver
        least_squares_solver=solver,
        # Ensure we don't carry over persistent state issues
        steady_state_initialization=True,
    )
    
    # Check that output exists and is finite
    state_ds = sim.io.load_dataset("state")
    assert state_ds is not None
    
    m_ind = state_ds["SH_m_ind"].values[-1]
    psi = state_ds["SH_psi"].values[-1]
    
    assert np.all(np.isfinite(m_ind))
    assert np.all(np.isfinite(psi))
    
    # Check norm is reasonable (not zero, not explosive)
    nm_mind = np.linalg.norm(m_ind)
    nm_psi = np.linalg.norm(psi)
    
    print(f"Solver {solver}: |m_ind|={nm_mind:.4e}, |psi|={nm_psi:.4e}")
    
    # Basic sanity bounds (based on typical IGRF responses)
    assert nm_mind > 1e-15
    assert nm_mind < 1e-3
    # psi can be zero if no toroidal driving force exists
    assert nm_psi < 1e-1
