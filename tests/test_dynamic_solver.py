"""Tests for the dynamic solver option."""

import pytest
import numpy as np
from pynamit.simulation.dynamics import SimulationMode
from pynamit.utils import to_numpy

def _run_dynamic_ramp_test(sim_mode, expected_psi_norm, expected_mind_norm, rel_tol=1e-4, test_name="sim"):
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
        Ncs=10,
        dynamics_mode="full_induction",
        simulation_mode=sim_mode.value,
        ignore_PFAC=False,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
    )

    # Verification
    ds = sim.io.load_dataset("state")

    assert ds is not None
    
    # Check evolution at t=1.0 (index 1)
    if "SH_psi" in ds.data_vars:
        psi_1 = ds["SH_psi"].values[1]
        m_ind_1 = ds["SH_m_ind"].values[1]
    elif "CS_psi" in ds.data_vars:
        psi_1 = ds["CS_psi"].values[1]
        m_ind_1 = ds["CS_m_ind"].values[1]
    elif "psi" in ds.data_vars:
         # Fallback for raw variable names
         psi_1 = ds["psi"].values[1]
         m_ind_1 = ds["m_ind"].values[1]
    else:
        pytest.fail("Neither SH_psi nor CS_psi found in dataset.")
    
    actual_psi_norm = np.linalg.norm(psi_1)
    actual_mind_norm = np.linalg.norm(m_ind_1)
    
    assert actual_psi_norm == pytest.approx(expected_psi_norm, rel=rel_tol)
    assert actual_mind_norm == pytest.approx(expected_mind_norm, rel=rel_tol)


def test_dynamic_ramp_pure_spectral():
    """Test pure spectral mode with dual induction."""
    # Baseline values for dual induction @ t=1.0s
    expected_psi_norm = 6.62065306935731e-08
    expected_mind_norm = 3.203992935564e-09
    
    _run_dynamic_ramp_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-10,
        test_name="sim_pure_spectral"
    )

def test_dynamic_ramp_spectral_transform_gl():
    """Test spectral transform GL mode with dual induction."""
    # Baseline values for dual induction @ t=1.0s
    expected_psi_norm = 6.62065306935731e-08
    expected_mind_norm = 3.959375316445736e-09
    
    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-10,
        test_name="sim_st_gl"
    )


def test_dynamic_ramp_spectral_transform_cs():
    """Test spectral transform CS mode with dual induction."""
    # Mode-specific baseline values for dual induction @ t=1.0s
    # Note: CS transform uses pseudo-inverse approximation, so values differ from pure spectral
    expected_psi_norm = 6.225616393348073e-08
    expected_mind_norm = 2.0694833942741078e-09

    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-10,
        test_name="sim_st_cs"
    )


def test_dynamic_ramp_cs_dominant():
    """Test CS dominant mode with dual induction."""
    # Mode-specific baseline values for dual induction @ t=1.0s
    # CS Dominant uses finite differences on cubed sphere.
    # Updated: Use lstsq-based minimum-norm steady state for numerical stability.
    # Previous value (3.015e-06) included null-space component that differed between backends.
    expected_psi_norm = 3.5008192104888524e-07 # Consistent
    expected_mind_norm = 0.00029393360233666624 # Minimum-norm solution (consistent across backends)

    print(f"DEBUG: CS Dominant Test - sim_mode={SimulationMode.CS_DOMINANT.value}")

    _run_dynamic_ramp_test(
        SimulationMode.CS_DOMINANT,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-5,
        test_name="sim_cs_dom",
        northern_apex_constraints=True
    )

def _run_dynamic_ramp_test(sim_mode, expected_psi_norm, expected_mind_norm, rel_tol=1e-4, test_name="sim", northern_apex_constraints=False):
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
        Ncs=10,
        dynamics_mode="full_induction",
        simulation_mode=sim_mode.value,
        ignore_PFAC=False,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
        northern_hemisphere_apex_constraints=northern_apex_constraints
    )

    # Verification
    ds = sim.io.load_dataset("state")

    assert ds is not None
    
    # Check evolution at t=1.0 (index 1)
    if "SH_psi" in ds.data_vars:
        psi_1 = ds["SH_psi"].values[1]
        m_ind_1 = ds["SH_m_ind"].values[1]
    elif "CS_psi" in ds.data_vars:
        psi_1 = ds["CS_psi"].values[1]
        m_ind_1 = ds["CS_m_ind"].values[1]
    elif "psi" in ds.data_vars:
         # Fallback for raw variable names
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
