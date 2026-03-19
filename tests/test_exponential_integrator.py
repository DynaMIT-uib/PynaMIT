"""Tests for the exponential integrator.

This test ensures that the matrix densification path (build_induction_matrix)
continues to work correctly. The exponential integrator requires the full
dense induction matrix for the steady-state form, so this serves as a
regression guard for the densification logic.

Uses 'full_induction' dynamics mode with exponential integrator.
"""

import pytest
import numpy as np
from pynamit.simulation.settings import SimulationMode
from pynamit.simulation.settings import (
    DynamicsMode,
    ExponentialSolverKind,
    IntegratorKind,
    MainfieldKind,
)


def _run_exponential_integrator_test(
    sim_mode,
    expected_psi_norm,
    expected_mind_norm,
    expected_steady_psi_norm=None,
    expected_steady_mind_norm=None,
    rel_tol=1e-4,
    test_name="sim",
    northern_apex_constraints=False,
    run_directory=None,
):
    """Helper to run the exponential integrator test using run_pynamit defaults."""
    from pynamit.simulation.runner import run_pynamit

    # Run using the default runner with multi_data=True
    # Using 'svd' solver for maximum precision
    sim = run_pynamit(
        run_directory=run_directory,
        final_time=2.0,
        dt=1.0,
        plotsteps=1,
        Nmax=10,
        Mmax=5,
        Ncs=10,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=sim_mode.value,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
        integrator=IntegratorKind.EXPONENTIAL,
        dense_full_operators=True,
        exponential_solver=ExponentialSolverKind.EXPM,
        northern_hemisphere_apex_constraints=northern_apex_constraints,
    )

    # Verification
    ds = sim.io.load_dataset("state")
    assert ds is not None
    steady_ds = sim.io.load_dataset("steady_state")
    assert steady_ds is not None

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
    if "SH_psi" in steady_ds.data_vars:
        psi_ss_1 = steady_ds["SH_psi"].values[1]
        m_ind_ss_1 = steady_ds["SH_m_ind"].values[1]
    elif "CS_psi" in steady_ds.data_vars:
        psi_ss_1 = steady_ds["CS_psi"].values[1]
        m_ind_ss_1 = steady_ds["CS_m_ind"].values[1]
    elif "psi" in steady_ds.data_vars:
        psi_ss_1 = steady_ds["psi"].values[1]
        m_ind_ss_1 = steady_ds["m_ind"].values[1]
    else:
        pytest.fail("Neither SH_psi nor CS_psi found in steady_state dataset.")

    actual_steady_psi_norm = np.linalg.norm(psi_ss_1)
    actual_steady_mind_norm = np.linalg.norm(m_ind_ss_1)

    print(f"DEBUG: {test_name} - Actual Psi={actual_psi_norm}, Expected Psi={expected_psi_norm}")
    print(
        f"DEBUG: {test_name} - Actual Mind={actual_mind_norm}, Expected Mind={expected_mind_norm}"
    )

    assert actual_psi_norm == pytest.approx(expected_psi_norm, rel=rel_tol)
    assert actual_mind_norm == pytest.approx(expected_mind_norm, rel=rel_tol)
    if expected_steady_psi_norm is not None:
        assert actual_steady_psi_norm == pytest.approx(expected_steady_psi_norm, rel=rel_tol)
    if expected_steady_mind_norm is not None:
        assert actual_steady_mind_norm == pytest.approx(expected_steady_mind_norm, rel=rel_tol)


def test_exponential_integrator_pure_spectral(tmp_path):
    """Test exponential integrator with pure spectral mode.

    This verifies the matrix densification path works for SH basis.
    """
    # Baselines updated 2026-03-11 after restoring the affine full-induction
    # exponential step. The saved steady state is kept as a separate diagnostic
    # and is not used as the step center in this multi-data case.
    expected_psi_norm = 1.0385777162078207e-08
    expected_mind_norm = 2.5031827572287588e-09
    expected_steady_psi_norm = 1.8757839161180914e-09
    expected_steady_mind_norm = 3.852979750457087e-09

    _run_exponential_integrator_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        expected_steady_psi_norm=expected_steady_psi_norm,
        expected_steady_mind_norm=expected_steady_mind_norm,
        rel_tol=1e-4,  # Relaxed slightly for physics changes
        test_name="exp_pure_spectral",
        run_directory=str(tmp_path / "exp_pure_spectral"),
    )


def test_exponential_integrator_spectral_transform_gl(tmp_path):
    """Test exponential integrator with spectral_transform (GL grid)."""
    expected_psi_norm = 1.0295375760679248e-08
    expected_mind_norm = 2.383869071166071e-09
    expected_steady_psi_norm = 1.5500692870120035e-09
    expected_steady_mind_norm = 3.683288112243564e-09

    _run_exponential_integrator_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        expected_steady_psi_norm=expected_steady_psi_norm,
        expected_steady_mind_norm=expected_steady_mind_norm,
        rel_tol=1e-4,
        test_name="exp_spec_trans_gl",
        run_directory=str(tmp_path / "exp_spec_trans_gl"),
    )


def test_exponential_integrator_spectral_transform_cs(tmp_path):
    """Test exponential integrator with spectral_transform (CS grid)."""
    expected_psi_norm = 6.707427162100316e-09
    expected_mind_norm = 1.5349727196517994e-09
    expected_steady_psi_norm = 1.2120821915824613e-09
    expected_steady_mind_norm = 2.3341514954809695e-09

    _run_exponential_integrator_test(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_psi_norm,
        expected_mind_norm,
        expected_steady_psi_norm=expected_steady_psi_norm,
        expected_steady_mind_norm=expected_steady_mind_norm,
        rel_tol=1e-4,
        test_name="exp_spec_trans_cs",
        run_directory=str(tmp_path / "exp_spec_trans_cs"),
    )


def test_exponential_integrator_cs_dominant(tmp_path):
    """Test exponential integrator with CS dominant mode.

    This verifies the matrix densification path works for CS basis
    with finite differences.
    """
    # Baselines updated 2026-03-19 after aligning CS tangential fields with the
    # active Helmholtz-potential representation and using the projected CS
    # toroidal electric potential in the induction forcing path. These values
    # are invariant under the repo cf/df sign choice.
    expected_psi_norm = 2.0905274555193079e-07
    expected_mind_norm = 8.02964670903966e-08
    expected_steady_psi_norm = 1.967846253931606e-07
    expected_steady_mind_norm = 1.203922898436875e-07

    _run_exponential_integrator_test(
        SimulationMode.CS_DOMINANT,
        expected_psi_norm,
        expected_mind_norm,
        expected_steady_psi_norm=expected_steady_psi_norm,
        expected_steady_mind_norm=expected_steady_mind_norm,
        rel_tol=1e-2,
        test_name="exp_cs_dom",
        northern_apex_constraints=True,
        run_directory=str(tmp_path / "exp_cs_dom"),
    )
