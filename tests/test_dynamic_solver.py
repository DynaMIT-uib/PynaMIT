"""Tests for the full-induction dynamic solver option."""

import pytest
import numpy as np

from pynamit.simulation.dynamics import SimulationMode


def _run_dynamic_ramp_test(
    sim_mode,
    expected_psi_norm,
    expected_mind_norm,
    rel_tol=1e-4,
    test_name="sim",
    northern_apex_constraints=False,
    filename_prefix=None,
):
    """Run full-induction simulation and compare t=1.0s state norms."""
    from pynamit.simulation.runner import run_pynamit

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
    # Baselines updated 2026-02-23 (pynamit-minimal) after unified weak-form Br
    # branch selector and shared Br-constrained jr->alpha mapping.
    expected_psi_norm = 1.855985373664705e-08
    expected_mind_norm = 4.511380698478858e-09

    _run_dynamic_ramp_test(
        SimulationMode.PURE_SPECTRAL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="sim_pure_spectral",
        filename_prefix=str(tmp_path / "sim_pure_spectral"),
    )


def test_dynamic_ramp_spectral_transform_gl(tmp_path):
    """Test spectral transform GL mode with dual induction."""
    # Baselines updated 2026-02-23 (pynamit-minimal) after unified weak-form Br
    # branch selector and shared Br-constrained jr->alpha mapping.
    expected_psi_norm = 1.861667931074767e-08
    expected_mind_norm = 4.923436484143163e-09

    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="sim_st_gl",
        filename_prefix=str(tmp_path / "sim_st_gl"),
    )


def test_dynamic_ramp_spectral_transform_cs(tmp_path):
    """Test spectral transform CS mode with dual induction."""
    # Baselines updated 2026-02-23 (pynamit-minimal) after unified weak-form Br
    # branch selector and shared Br-constrained jr->alpha mapping.
    expected_psi_norm = 8.022390477117027e-09
    expected_mind_norm = 1.9769085065985536e-09

    _run_dynamic_ramp_test(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=1e-4,
        test_name="sim_st_cs",
        filename_prefix=str(tmp_path / "sim_st_cs"),
    )


def test_dynamic_ramp_cs_dominant(backend: str, tmp_path):
    """Test CS dominant mode with dual induction."""
    # Baselines updated 2026-02-23 after coupled-operator refactor cleanup and
    # CS-dominant full-induction stabilization changes.
    expected_psi_norm = 2.481844448216916e-07
    expected_mind_norm = 2.2003977096218905e-07
    rel_tol = 1e-2

    _run_dynamic_ramp_test(
        SimulationMode.CS_DOMINANT,
        expected_psi_norm,
        expected_mind_norm,
        rel_tol=rel_tol,
        test_name="sim_cs_dom",
        northern_apex_constraints=True,
        filename_prefix=str(tmp_path / "sim_cs_dom"),
    )