"""Tests for the full-induction dynamic solver option."""

import pytest
import numpy as np

from pynamit.simulation.settings import SimulationMode
from pynamit.simulation.settings import DynamicsMode, MainfieldKind


def _run_dynamic_ramp_test(
    sim_mode,
    expected_psi_norm,
    expected_mind_norm,
    rel_tol=1e-4,
    test_name="sim",
    northern_apex_constraints=False,
    run_directory=None,
):
    """Run full-induction simulation and compare t=1.0s state norms."""
    from pynamit.simulation.runner import run_pynamit

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
        northern_hemisphere_apex_constraints=northern_apex_constraints,
    )

    ds = sim.io.load_dataset("state")
    assert ds is not None

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


def test_dynamic_ramp_pure_spectral(tmp_path):
    """Test pure spectral mode with dual induction."""
    # Baselines updated 2026-02-27 (pynamit-minimal) after full-induction
    # alpha-space HL/LL hard-constraint mapping.
    expected_psi_norm = 8.255493120634494e-09
    expected_mind_norm = 1.0382374674678734e-09

    _run_dynamic_ramp_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="sim_pure_spectral",
        run_directory=str(tmp_path / "sim_pure_spectral"),
    )


def test_dynamic_ramp_spectral_transform_gl(tmp_path):
    """Test spectral transform GL mode with dual induction."""
    # Baselines updated 2026-02-27 (pynamit-minimal) after full-induction
    # alpha-space HL/LL hard-constraint mapping.
    expected_psi_norm = 8.25732003096468e-09
    expected_mind_norm = 1.0576189485121287e-09

    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="sim_st_gl",
        run_directory=str(tmp_path / "sim_st_gl"),
    )


def test_dynamic_ramp_spectral_transform_cs(tmp_path):
    """Test spectral transform CS mode with dual induction."""
    # Baselines updated 2026-02-27 (pynamit-minimal) after full-induction
    # alpha-space HL/LL hard-constraint mapping.
    expected_psi_norm = 4.1153778728659436e-09
    expected_mind_norm = 9.87049993973558e-10

    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="sim_st_cs",
        run_directory=str(tmp_path / "sim_st_cs"),
    )


def test_dynamic_ramp_cs_dominant(backend: str, tmp_path):
    """Test CS dominant mode with dual induction."""
    # Baselines updated 2026-02-27 (pynamit-minimal) after full-induction
    # alpha-space HL/LL hard-constraint mapping.
    expected_psi_norm = 2.0203700322434142e-07
    expected_mind_norm = 1.9512331416159726e-07
    rel_tol = 1e-2

    _run_dynamic_ramp_test(
        SimulationMode.CS_DOMINANT,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=rel_tol,
        test_name="sim_cs_dom",
        northern_apex_constraints=True,
        run_directory=str(tmp_path / "sim_cs_dom"),
    )
