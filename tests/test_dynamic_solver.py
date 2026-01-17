"""Tests for the dynamic solver option."""

import pytest
import numpy as np
from pynamit.simulation.dynamics import SimulationMode
from pynamit.utils import to_numpy

def _run_dynamic_ramp_test(sim_mode, expected_psi_norm, expected_mind_norm, rel_tol=1e-4):
    """Helper to run the dual induction test using run_pynamit defaults."""
    from pynamit.simulation.runner import run_pynamit
    
    # Run using the default runner with multi_data=True
    # Using 'direct' solver for maximum precision and tighter tolerances
    sim = run_pynamit(
        final_time=2.0,
        dt=1.0,
        plotsteps=1, 
        Nmax=10,
        Mmax=5,
        dynamics_mode="full_induction",
        simulation_mode=sim_mode.value,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
    )

    # Verification
    ds = sim.io.load_dataset("state")
    assert ds is not None
    assert "SH_psi" in ds.data_vars
    assert "SH_m_ind" in ds.data_vars
    
    # Check evolution at t=1.0 (index 1)
    psi_1 = ds["SH_psi"].values[1]
    m_ind_1 = ds["SH_m_ind"].values[1]
    
    actual_psi_norm = np.linalg.norm(psi_1)
    actual_mind_norm = np.linalg.norm(m_ind_1)
    
    assert actual_psi_norm == pytest.approx(expected_psi_norm, rel=rel_tol)
    assert actual_mind_norm == pytest.approx(expected_mind_norm, rel=rel_tol)


def test_dynamic_ramp_pure_spectral():
    """Test pure spectral mode with dual induction."""
    # Baseline values for dual induction @ t=1.0s
    expected_psi_norm = 6.62065307e-11
    expected_mind_norm = 3.20399294e-09
    
    _run_dynamic_ramp_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-5
    )

def test_dynamic_ramp_spectral_transform_gl():
    """Test spectral transform GL mode with dual induction."""
    # Baseline values for dual induction @ t=1.0s
    expected_psi_norm = 6.62065307e-11
    expected_mind_norm = 3.20014482e-09
    
    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-5
    )
