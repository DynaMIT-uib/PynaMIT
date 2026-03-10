"""Regression tests for magnetospheric toroidal/poloidal lock options."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.math.constants import RE, mu0
from pynamit.simulation.dynamics import Dynamics, SimulationMode
from pynamit.simulation.spatial import to_dense
from pynamit.simulation.settings import DynamicsMode, MainfieldKind


def _build_dynamics(
    tmp_path,
    *,
    toroidal_lock: bool,
    poloidal_lock: bool,
) -> Dynamics:
    return Dynamics(
        run_directory=str(
            tmp_path
            / f"locks_t{int(toroidal_lock)}_p{int(poloidal_lock)}"
        ),
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
    """Verify lock toggles wire into hard constraints and RM coupling operators."""
    dynamics = _build_dynamics(
        tmp_path,
        toroidal_lock=toroidal_lock,
        poloidal_lock=poloidal_lock,
    )
    state = dynamics.state
    geometry = state.geometry
    basis = state.basis
    n_coeffs = basis.index_length

    # --- Toroidal lock: optional HL hard rows in constraint bundle ---
    bundle = state.constraints.induction_constraint_bundle_hard
    assert bundle is not None
    assert bundle["C_ll"].shape[1] == n_coeffs
    assert bundle["C_hl"].shape[1] == n_coeffs
    assert bundle["C_total"].shape[1] == n_coeffs
    assert bundle["C_total"].shape[0] == bundle["C_ll"].shape[0] + bundle["C_hl"].shape[0]
    if toroidal_lock:
        assert bundle["C_hl"].shape[0] > 0
    else:
        assert bundle["C_hl"].shape[0] == 0

    # --- Poloidal lock: RM coupling on induced pathways; imposed Br is always closed ---
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

    br_rm_to_ri_shift, br_ri_to_rm_shift, rm_roundtrip_denominator = geometry._pfac.get_coupling_factors()
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
        if poloidal_lock
        else np.asarray(state.poloidal_matrices.T_to_Ve_open)
    )

    np.testing.assert_allclose(p_psi, expected_p_psi, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(t_psi, expected_t_psi, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("poloidal_lock", [False, True])
def test_legacy_keeps_toroidal_source_pfac_baseline(tmp_path, poloidal_lock: bool) -> None:
    """Legacy mode keeps m_imp locked while psi channel remains unlocked."""
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


@pytest.mark.parametrize("poloidal_lock", [False, True])
def test_full_induction_lock_numeric_snapshot(tmp_path, poloidal_lock: bool) -> None:
    """Numeric regression snapshot for lock semantics (full-induction, RM=4*RE)."""
    dynamics = _build_dynamics(
        tmp_path,
        toroidal_lock=False,
        poloidal_lock=poloidal_lock,
    )
    state = dynamics.state
    geometry = state.geometry
    n_coeffs = state.basis.index_length

    bundle = state.constraints.induction_constraint_bundle_hard
    assert bundle is not None
    # LL hard constraints are expressed in mismatch space projected to LL modes.
    assert bundle["C_ll"].shape[0] > 0
    assert bundle["C_ll"].shape[1] == n_coeffs
    assert bundle["C_hl"].shape[0] == 0

    op_m_ind = geometry.get_potential_to_JS_operator("m_ind", mode=None)
    op_br = geometry.get_potential_to_JS_operator("Br", mode=None)
    op_m_imp = geometry.get_potential_to_JS_operator("m_imp", mode=None)
    op_psi = geometry.get_potential_to_JS_operator("psi", mode=None)

    t_m_ind = np.asarray(to_dense(op_m_ind))[n_coeffs:, :]
    t_br = np.asarray(to_dense(op_br))[n_coeffs:, :]
    t_m_imp = np.asarray(to_dense(op_m_imp))[n_coeffs:, :]
    t_psi = np.asarray(to_dense(op_psi))[n_coeffs:, :]

    expected = {
        False: {
            "norm_t_m_ind": 73193972.32756677,
            "norm_t_br": 2137241.9153284496,
            "norm_t_m_imp": 0.7223795135300719,
            "norm_t_psi": 0.7253300455734074,
        },
        True: {
            "norm_t_m_ind": 73199298.94320045,
            "norm_t_br": 2137241.9153284496,
            "norm_t_m_imp": 0.7223795135300719,
            "norm_t_psi": 0.7162627561888577,
        },
    }[poloidal_lock]

    assert float(np.linalg.norm(t_m_ind)) == pytest.approx(expected["norm_t_m_ind"], rel=1e-10)
    assert float(np.linalg.norm(t_br)) == pytest.approx(expected["norm_t_br"], rel=1e-10)
    assert float(np.linalg.norm(t_m_imp)) == pytest.approx(expected["norm_t_m_imp"], rel=1e-12)
    assert float(np.linalg.norm(t_psi)) == pytest.approx(expected["norm_t_psi"], rel=1e-12)
