"""Tests for the exponential integrator.

This test ensures that the matrix densification path (build_induction_matrix)
continues to work correctly. The exponential integrator requires the full
dense induction matrix for the steady-state form, so this serves as a 
regression guard for the densification logic.

Uses 'full_induction' dynamics mode with exponential integrator.
"""

import pytest
import numpy as np
from pynamit.simulation.dynamics import SimulationMode


def _run_exponential_integrator_test(
    sim_mode, 
    expected_psi_norm,
    expected_mind_norm, 
    rel_tol=1e-4, 
    test_name="sim",
    northern_apex_constraints=False,
    filename_prefix=None,
):
    """Helper to run the exponential integrator test using run_pynamit defaults."""
    from pynamit.simulation.runner import run_pynamit
    
    # Run using the default runner with multi_data=True
    # Using 'svd' solver for maximum precision
    sim = run_pynamit(
        filename_prefix=filename_prefix,
        final_time=2.0,
        dt=1.0,
        plotsteps=1, 
        Nmax=10,
        Mmax=5,
        Ncs=10,
        dynamics_mode="full_induction",
        simulation_mode=sim_mode.value,
        ignore_PFAC=False,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
        integrator="exponential",
        northern_hemisphere_apex_constraints=northern_apex_constraints,
    )

    # Verification
    ds = sim.io.load_dataset("state")
    assert ds is not None
    
    # Check evolution at t=1.0 (index 1)
    # Full induction mode has both psi and m_ind
    if "SH_psi" in ds.data_vars:
        psi_1 = ds["SH_psi"].values[1]
        m_ind_1 = ds["SH_m_ind"].values[1]
    elif "CS_psi" in ds.data_vars:
        psi_1 = ds["CS_psi"].values[1]
        m_ind_1 = ds["CS_m_ind"].values[1]
    elif "psi" in ds.data_vars:
        psi_1 = ds["psi"].values[1]
        m_ind_1 = ds["m_ind"].values[1]
    else:
        pytest.fail("Neither SH_psi nor CS_psi found in dataset.")
    
    actual_psi_norm = np.linalg.norm(psi_1)
    actual_mind_norm = np.linalg.norm(m_ind_1)
    
    print(f"DEBUG: {test_name} - Actual Psi={actual_psi_norm}, Expected Psi={expected_psi_norm}")
    print(f"DEBUG: {test_name} - Actual Mind={actual_mind_norm}, Expected Mind={expected_mind_norm}")

    assert actual_psi_norm == pytest.approx(expected_psi_norm, rel=rel_tol)
    assert actual_mind_norm == pytest.approx(expected_mind_norm, rel=rel_tol)


def test_exponential_integrator_pure_spectral(tmp_path):
    """Test exponential integrator with pure spectral mode.
    
    This verifies the matrix densification path works for SH basis.
    """
    # Baselines updated 2026-02-23 (pynamit-minimal).
    expected_psi_norm = 1.666062771592729e-08
    expected_mind_norm = 3.732546017407569e-09
    
    _run_exponential_integrator_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,  # Relaxed slightly for physics changes
        test_name="exp_pure_spectral",
        filename_prefix=str(tmp_path / "exp_pure_spectral"),
    )

def test_exponential_integrator_spectral_transform_gl(tmp_path):
    """Test exponential integrator with spectral_transform (GL grid)."""
    # Baselines updated 2026-02-23 (pynamit-minimal).
    expected_psi_norm = 1.6688556536813525e-08
    expected_mind_norm = 4.163678599947615e-09
    
    _run_exponential_integrator_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="exp_spec_trans_gl",
        filename_prefix=str(tmp_path / "exp_spec_trans_gl"),
    )

def test_exponential_integrator_spectral_transform_cs(tmp_path):
    """Test exponential integrator with spectral_transform (CS grid)."""
    # Baselines updated 2026-02-23 (pynamit-minimal).
    expected_psi_norm = 8.668242382007195e-09
    expected_mind_norm = 1.4815412431438688e-09

    _run_exponential_integrator_test(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="exp_spec_trans_cs",
        filename_prefix=str(tmp_path / "exp_spec_trans_cs"),
    )

def test_exponential_integrator_cs_dominant(tmp_path):
    """Test exponential integrator with CS dominant mode.

    This verifies the matrix densification path works for CS basis
    with finite differences.
    """
    # Baselines updated 2026-02-23 after coupled-operator refactor cleanup and
    # CS-dominant full-induction stabilization changes.
    expected_psi_norm = 2.481844448216916e-07
    expected_mind_norm = 2.1994620444111382e-07

    _run_exponential_integrator_test(
        SimulationMode.CS_DOMINANT,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-2,
        test_name="exp_cs_dom",
        northern_apex_constraints=True,
        filename_prefix=str(tmp_path / "exp_cs_dom"),
    )
