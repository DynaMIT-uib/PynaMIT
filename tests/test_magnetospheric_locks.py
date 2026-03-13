"""Regression tests for magnetospheric toroidal/poloidal lock options."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.math.constants import RE, mu0
from pynamit.simulation.dynamics import Dynamics, SimulationMode
from pynamit.simulation.spatial import to_dense
from pynamit.simulation.settings import DynamicsMode, MainfieldKind


def _build_dynamics(tmp_path, *, toroidal_lock: bool, poloidal_lock: bool) -> Dynamics:
    return Dynamics(
        run_directory=str(tmp_path / f"locks_t{int(toroidal_lock)}_p{int(poloidal_lock)}"),
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
        magnetospheric_toroidal_lock=toroidal_lock,
        magnetospheric_poloidal_lock=poloidal_lock,
    )


def _build_legacy_dynamics(tmp_path, *, poloidal_lock: bool) -> Dynamics:
    return Dynamics(
        run_directory=str(tmp_path / f"legacy_lock_p{int(poloidal_lock)}"),
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
        magnetospheric_poloidal_lock=poloidal_lock,
    )


@pytest.mark.parametrize("toroidal_lock", [False, True])
@pytest.mark.parametrize("poloidal_lock", [False, True])
def test_full_induction_magnetospheric_locks(
    tmp_path, toroidal_lock: bool, poloidal_lock: bool
) -> None:
    """Verify lock toggles wire into LL constraints and RM coupling operators."""
    dynamics = _build_dynamics(tmp_path, toroidal_lock=toroidal_lock, poloidal_lock=poloidal_lock)
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

    # --- Boundary locks: poloidal lock affects induced poloidal/FAC pathways
    # --- from ``m_ind``/``Br``, while toroidal lock affects the dynamic
    # --- toroidal/FAC PFAC reaction of ``psi``. Imposed RM driver channels
    # --- remain closed.
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
    if poloidal_lock:
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
    expected_t_psi = (
        np.asarray(state.poloidal_matrices.T_to_Ve)
        if toroidal_lock
        else np.asarray(state.poloidal_matrices.T_to_Ve_open)
    )

    np.testing.assert_allclose(p_psi, expected_p_psi, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(t_psi, expected_t_psi, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("poloidal_lock", [False, True])
def test_legacy_keeps_toroidal_source_pfac_baseline(tmp_path, poloidal_lock: bool) -> None:
    """Legacy mode keeps the imposed RM-closed source baseline; dynamic psi stays open."""
    dynamics = _build_legacy_dynamics(tmp_path, poloidal_lock=poloidal_lock)
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
    expected_t_psi = np.asarray(state.poloidal_matrices.T_to_Ve_open)
    np.testing.assert_allclose(t_m_imp, expected_t_m_imp, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(t_psi, expected_t_psi, atol=1e-10, rtol=1e-10)


def test_full_induction_toroidal_lock_keeps_dynamic_dtpsi_runtime_open(tmp_path) -> None:
    """Toroidal lock should not alter the live shell ``dt_psi`` operators."""
    dyn_open = _build_dynamics(tmp_path, toroidal_lock=False, poloidal_lock=False)
    dyn_closed = _build_dynamics(tmp_path, toroidal_lock=True, poloidal_lock=False)

    mats_open = dyn_open.state.toroidal_matrices
    mats_closed = dyn_closed.state.toroidal_matrices
    assert mats_open is not None
    assert mats_closed is not None

    alpha_to_psi_open = np.asarray(mats_open.alpha_to_psi_coeff_operator)
    alpha_to_psi_closed = np.asarray(mats_closed.alpha_to_psi_coeff_operator)
    radial_open = np.asarray(mats_open.radial_closure_dt_psi_from_dtalpha)
    radial_closed = np.asarray(mats_closed.radial_closure_dt_psi_from_dtalpha)

    np.testing.assert_allclose(
        alpha_to_psi_closed,
        alpha_to_psi_open,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        radial_closed,
        radial_open,
        atol=1e-10,
        rtol=1e-10,
    )


def test_toroidal_rm_reaction_prototype_exposes_runtime_and_boundary_operator(tmp_path) -> None:
    """RM prototype should expose runtime-open and explicit boundary operators."""
    dynamics = _build_dynamics(tmp_path, toroidal_lock=True, poloidal_lock=False)
    proto = dynamics.state.toroidal_rm_reaction_prototype
    report = dynamics.state.get_toroidal_rm_reaction_report()

    np.testing.assert_allclose(
        proto.alpha_to_psi_closed,
        proto.alpha_to_psi_open + proto.alpha_to_psi_reaction,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.radial_closure_dt_psi_closed,
        proto.radial_closure_dt_psi_open + proto.radial_closure_dt_psi_reaction,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.toroidal_feedback_dtalpha_closed,
        proto.toroidal_feedback_dtalpha_open + proto.toroidal_feedback_dtalpha_reaction,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.dynamic_pfac_closed,
        proto.dynamic_pfac_open + proto.dynamic_pfac_reaction,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.alpha_to_psi_closed,
        proto.alpha_to_psi_open,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.radial_closure_dt_psi_closed,
        proto.radial_closure_dt_psi_open,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.closure_denominator @ proto.alpha_to_psi_shell_closed,
        proto.alpha_to_psi_open,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.alpha_to_psi_shell_closed,
        proto.alpha_to_psi_open + (proto.roundtrip_gain @ proto.alpha_to_psi_shell_closed),
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        proto.alpha_to_psi_shell_closed - proto.alpha_to_psi_open,
        proto.shell_reaction_operator @ proto.alpha_to_psi_open,
        atol=1e-10,
        rtol=1e-10,
    )
    assert np.linalg.norm(proto.alpha_to_psi_reaction) < 1e-12
    assert np.linalg.norm(proto.alpha_to_psi_shell_closed - proto.alpha_to_psi_closed) > 0.0
    assert np.linalg.norm(proto.alpha_to_normal_current_rm_grid) > 0.0
    assert np.linalg.norm(proto.alpha_to_divergent_closure_current_rm_grid) > 0.0
    assert np.linalg.norm(proto.alpha_to_sheet_boundary_psi_rm) > 0.0
    assert np.linalg.norm(proto.alpha_to_dynamic_pfac_reaction) > 0.0

    assert report["shell_boundary_closure"]["fixed_point_residual_norm"] < 1e-12
    assert report["shell_boundary_closure"]["denominator_residual_norm"] < 1e-12
    assert report["shell_boundary_closure"]["reaction_operator_residual_norm"] < 1e-12
    assert report["shell_boundary_closure"]["sheet_rm_value_mismatch_norm"] > 0.0
    assert report["shell_boundary_closure"]["runtime_vs_shell_closed_mismatch_norm"] > 0.0
    assert report["shell_boundary_closure"]["runtime_vs_shell_radial_mismatch_norm"] > 0.0
    assert report["alpha_to_psi"]["reaction_norm"] < 1e-12
    assert report["rm_boundary_closure"]["normal_current_operator_norm"] > 0.0
    assert report["dynamic_pfac"]["alpha_reaction_norm"] > 0.0
    assert report["alpha_to_psi"]["closure_residual_norm"] < 1e-12
