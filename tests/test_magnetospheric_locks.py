"""Regression tests for magnetospheric shielding and RM boundary diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.math.constants import RE, mu0
from pynamit.simulation.dynamics import Dynamics, SimulationMode
from pynamit.simulation.spatial import to_dense
from pynamit.simulation.settings import DynamicsMode, MainfieldKind


def _build_dynamics(tmp_path, *, shielding: bool) -> Dynamics:
    return Dynamics(
        run_directory=str(tmp_path / f"locks_p{int(shielding)}"),
        Nmax=8,
        Mmax=4,
        Ncs=10,
        RI=RE + 110.0e3,
        RM=4.0 * RE,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        connect_hemispheres=True,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
        magnetospheric_shielding=shielding,
    )


def _build_legacy_dynamics(tmp_path, *, shielding: bool) -> Dynamics:
    return Dynamics(
        run_directory=str(tmp_path / f"legacy_lock_p{int(shielding)}"),
        Nmax=8,
        Mmax=4,
        Ncs=10,
        RI=RE + 110.0e3,
        RM=4.0 * RE,
        mainfield_kind=MainfieldKind.DIPOLE,
        ignore_PFAC=False,
        connect_hemispheres=True,
        dynamics_mode=DynamicsMode.LEGACY,
        simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
        magnetospheric_shielding=shielding,
    )


@pytest.mark.parametrize("shielding", [False, True])
def test_full_induction_magnetospheric_shielding(tmp_path, shielding: bool) -> None:
    """Verify shielding toggles wire into LL constraints and RM coupling operators."""
    dynamics = _build_dynamics(tmp_path, shielding=shielding)
    state = dynamics.state
    geometry = state.geometry
    basis = state.basis
    n_coeffs = basis.index_length

    # --- LL constraints are independent of toroidal RM boundary lock ---
    bundle = state.constraints.induction_constraint_bundle_hard
    assert bundle is not None
    assert bundle["C_ll"].shape[1] == n_coeffs
    assert bundle["C_total"].shape[1] == n_coeffs
    assert bundle["C_total"].shape[0] == bundle["C_ll"].shape[0]
    assert bundle["C_ll"].shape[0] > 0

    # --- Magnetospheric shielding affects induced poloidal/FAC pathways from
    # --- ``m_ind``/``Br`` and the dynamic ``psi -> Ve`` response, while the
    # --- toroidal boundary object remains diagnostic/interface-only. Imposed
    # --- RM driver channels remain closed.
    op_m_ind = geometry.get_potential_to_JS_operator("m_ind", mode=None)
    op_br = geometry.get_potential_to_JS_operator("Br", mode=None)
    op_m_imp = geometry.get_potential_to_JS_operator("m_imp", mode=None)
    op_psi = geometry.get_potential_to_JS_operator("psi", mode=None)
    dense_m_ind = np.asarray(to_dense(op_m_ind))
    dense_br = np.asarray(to_dense(op_br))
    dense_m_imp = np.asarray(to_dense(op_m_imp))
    dense_psi = np.asarray(to_dense(op_psi))

    t_m_ind = dense_m_ind[n_coeffs:, :]
    p_m_ind = dense_m_ind[:n_coeffs, :]
    t_br = dense_br[n_coeffs:, :]
    p_br = dense_br[:n_coeffs, :]
    t_m_imp = dense_m_imp[n_coeffs:, :]
    p_m_imp = dense_m_imp[:n_coeffs, :]
    t_psi = dense_psi[n_coeffs:, :]
    p_psi = dense_psi[:n_coeffs, :]

    scaling = np.asarray(to_dense(basis.get_potential_scaling_operator()))
    expected_t_m_ind = (-1.0 / mu0) * scaling

    br_rm_to_ri_shift, br_ri_to_rm_shift, rm_roundtrip_denominator = (
        geometry._pfac.get_coupling_factors()
    )
    rm_feedback_term = (br_rm_to_ri_shift * br_ri_to_rm_shift) / rm_roundtrip_denominator
    if shielding:
        expected_t_m_ind = expected_t_m_ind * (1.0 + rm_feedback_term)

    np.testing.assert_allclose(p_m_ind, 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(t_m_ind, expected_t_m_ind, atol=1e-10, rtol=1e-10)

    lap_diag = np.diag(np.asarray(to_dense(basis.get_laplacian_operator(state.RI))))
    m_ind_to_br = -(state.RI**2) * lap_diag
    br_factor = -br_rm_to_ri_shift / rm_roundtrip_denominator
    expected_t_br = (-1.0 / mu0) * scaling * (br_factor / m_ind_to_br)[:, None]

    np.testing.assert_allclose(p_br, 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(t_br, expected_t_br, atol=1e-10, rtol=1e-10)

    expected_p_m_imp = (1.0 / mu0) * np.eye(n_coeffs)
    expected_t_m_imp = np.asarray(state.poloidal_matrices.T_to_Ve)
    expected_t_m_imp = (1.0 / rm_roundtrip_denominator)[:, None] * expected_t_m_imp

    np.testing.assert_allclose(p_m_imp, expected_p_m_imp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(t_m_imp, expected_t_m_imp, atol=1e-10, rtol=1e-10)

    expected_p_psi = (1.0 / mu0) * np.eye(n_coeffs)
    if shielding:
        expected_t_psi = np.asarray(state.poloidal_matrices.T_to_Ve)
    else:
        expected_t_psi = np.asarray(state.poloidal_matrices.T_to_Ve_open)

    np.testing.assert_allclose(p_psi, expected_p_psi, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(t_psi, expected_t_psi, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("shielding", [False, True])
def test_legacy_keeps_toroidal_source_pfac_baseline(tmp_path, shielding: bool) -> None:
    """Legacy mode keeps imposed RM-closed source baseline; dynamic ``psi`` follows shielding."""
    dynamics = _build_legacy_dynamics(tmp_path, shielding=shielding)
    state = dynamics.state
    geometry = state.geometry
    n_coeffs = state.basis.index_length

    op_m_imp = geometry.get_potential_to_JS_operator("m_imp", mode=None)
    op_psi = geometry.get_potential_to_JS_operator("psi", mode=None)
    dense_m_imp = np.asarray(to_dense(op_m_imp))
    dense_psi = np.asarray(to_dense(op_psi))
    t_m_imp = dense_m_imp[n_coeffs:, :]
    t_psi = dense_psi[n_coeffs:, :]
    expected_t_m_imp = np.asarray(state.poloidal_matrices.T_to_Ve)
    _, _, rm_roundtrip_denominator = geometry._pfac.get_coupling_factors()
    expected_t_m_imp = (1.0 / rm_roundtrip_denominator)[:, None] * expected_t_m_imp
    if shielding:
        expected_t_psi = np.asarray(state.poloidal_matrices.T_to_Ve)
    else:
        expected_t_psi = np.asarray(state.poloidal_matrices.T_to_Ve_open)
    np.testing.assert_allclose(t_m_imp, expected_t_m_imp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(t_psi, expected_t_psi, atol=1e-10, rtol=1e-10)


def test_toroidal_rm_boundary_diagnostics_expose_runtime_and_boundary_operator(tmp_path) -> None:
    """Runtime stays open while explicit ``R_M`` boundary diagnostics remain available."""
    dynamics = _build_dynamics(tmp_path, shielding=False)
    state = dynamics.state
    boundary_source = state.toroidal_rm_boundary_operators
    rm_ops = state.poloidal_matrices.toroidal_rm_closure_operators

    assert np.linalg.norm(np.asarray(boundary_source.alpha_to_boundary_psi_rm)) > 0.0
    assert (
        np.linalg.norm(np.asarray(state.poloidal_matrices.dynamic_toroidal_pfac_reaction_operator))
        < 1e-12
    )
    assert np.linalg.norm(np.asarray(rm_ops.alpha_to_normal_current_rm_grid)) > 0.0
    assert np.linalg.norm(np.asarray(rm_ops.alpha_to_closure_potential_rm_coeff)) > 0.0
    assert np.linalg.norm(np.asarray(rm_ops.alpha_to_divergent_closure_current_rm_grid)) > 0.0


def test_shielding_closes_dynamic_psi_pfac_path(tmp_path) -> None:
    """Magnetospheric shielding should close the dynamic ``psi -> Ve`` PFAC pathway."""
    dyn_open = _build_dynamics(tmp_path, shielding=False)
    dyn_closed = _build_dynamics(tmp_path, shielding=True)

    state_open = dyn_open.state
    state_closed = dyn_closed.state

    np.testing.assert_allclose(
        np.asarray(state_open.poloidal_matrices.dynamic_toroidal_pfac_reaction_operator),
        0.0,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(state_closed.poloidal_matrices.dynamic_toroidal_pfac_closed_operator),
        np.asarray(state_closed.poloidal_matrices.T_to_Ve),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(state_closed.poloidal_matrices.dynamic_toroidal_pfac_open_operator)
        + np.asarray(state_closed.poloidal_matrices.dynamic_toroidal_pfac_reaction_operator),
        np.asarray(state_closed.poloidal_matrices.dynamic_toroidal_pfac_closed_operator),
        atol=1e-10,
        rtol=1e-10,
    )
    assert (
        np.linalg.norm(
            np.asarray(state_closed.poloidal_matrices.dynamic_toroidal_pfac_reaction_operator)
        )
        > 0.0
    )


def test_magnetospheric_boundary_diagnostics_follow_shielding(tmp_path) -> None:
    """Induced boundary diagnostics should vanish above ``R_M`` when shielding is enabled."""
    dyn_open = _build_dynamics(tmp_path, shielding=False)
    dyn_closed = _build_dynamics(tmp_path, shielding=True)

    report_open = dyn_open.state.get_magnetospheric_boundary_report()
    report_closed = dyn_closed.state.get_magnetospheric_boundary_report()

    assert report_open["m_ind_to_br_rm"]["open_norm"] > 0.0
    assert report_open["m_ind_to_br_rm"]["effective_norm"] > 0.0
    assert report_open["m_ind_to_br_rm"]["shielding_norm"] < 1e-12
    assert report_open["dynamic_psi_to_ve_rm"]["open_norm"] > 0.0
    assert report_open["dynamic_psi_to_ve_rm"]["effective_norm"] > 0.0
    assert report_open["dynamic_psi_to_ve_rm"]["shielding_norm"] < 1e-12
    assert report_open["dynamic_alpha_to_psi_rm"]["open_norm"] > 0.0
    assert report_open["dynamic_alpha_to_psi_rm"]["effective_norm"] > 0.0
    assert report_open["dynamic_alpha_to_psi_rm"]["shielding_norm"] < 1e-12

    assert report_closed["m_ind_to_br_rm"]["open_norm"] > 0.0
    assert report_closed["m_ind_to_br_rm"]["effective_norm"] < 1e-12
    assert report_closed["m_ind_to_br_rm"]["shielding_norm"] > 0.0
    assert report_closed["dynamic_psi_to_ve_rm"]["open_norm"] > 0.0
    assert report_closed["dynamic_psi_to_ve_rm"]["effective_norm"] < 1e-12
    assert report_closed["dynamic_psi_to_ve_rm"]["shielding_norm"] > 0.0
    assert report_closed["dynamic_alpha_to_psi_rm"]["open_norm"] > 0.0
    assert report_closed["dynamic_alpha_to_psi_rm"]["effective_norm"] > 0.0
    assert report_closed["dynamic_alpha_to_psi_rm"]["shielding_norm"] < 1e-12
