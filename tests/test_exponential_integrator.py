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
    northern_apex_constraints=False
):
    """Helper to run the exponential integrator test using run_pynamit defaults."""
    from pynamit.simulation.runner import run_pynamit
    
    # Run using the default runner with multi_data=True
    # Using 'svd' solver for maximum precision
    sim = run_pynamit(
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


def test_exponential_integrator_pure_spectral():
    """Test exponential integrator with pure spectral mode.
    
    This verifies the matrix densification path works for SH basis.
    """
    # Baselines updated 2026-02-20 after affine expm stepping for full induction.
    expected_psi_norm = 1.7997413683476614e-08
    expected_mind_norm = 3.376627581325625e-09
    
    _run_exponential_integrator_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,  # Relaxed slightly for physics changes
        test_name="exp_pure_spectral"
    )

def test_exponential_integrator_spectral_transform_gl():
    """Test exponential integrator with spectral_transform (GL grid)."""
    # Baselines updated 2026-02-20 after affine expm stepping for full induction.
    expected_psi_norm = 1.8008887048492457e-08
    expected_mind_norm = 4.067094982748926e-09
    
    _run_exponential_integrator_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="exp_spec_trans_gl"
    )

def test_exponential_integrator_spectral_transform_cs():
    """Test exponential integrator with spectral_transform (CS grid)."""
    # Baselines updated 2026-02-20 after affine expm stepping for full induction.
    expected_psi_norm = 9.130515936293893e-09
    expected_mind_norm = 1.8220631527925785e-09

    _run_exponential_integrator_test(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="exp_spec_trans_cs"
    )

def test_exponential_integrator_cs_dominant():
    """Test exponential integrator with CS dominant mode.

    This verifies the matrix densification path works for CS basis
    with finite differences.
    """
    # Baselines updated 2026-02-20 after coupled-operator toroidal feedback
    # regularization split and fixed-step exponential integration checks.
    expected_psi_norm = 8.65753066917622e-08
    expected_mind_norm = 7.977838754768042e-08

    _run_exponential_integrator_test(
        SimulationMode.CS_DOMINANT,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-2,
        test_name="exp_cs_dom",
        northern_apex_constraints=True
    )
