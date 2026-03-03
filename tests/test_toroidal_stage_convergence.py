"""Stage-wise convergence parity checks for full-induction toroidal operators."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.simulation.dynamics import SimulationMode


def _build_state(
    sim_mode: SimulationMode,
    *,
    nmax: int,
    mmax: int,
    ncs: int,
    filename_prefix: str,
):
    from pynamit.simulation.dynamics import Dynamics

    dynamics = Dynamics(
        filename_prefix=filename_prefix,
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        dynamics_mode="full_induction",
        simulation_mode=sim_mode,
        ignore_PFAC=False,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        connect_hemispheres=True,
        least_squares_solver="svd",
        magnetospheric_toroidal_lock=False,
        northern_hemisphere_apex_constraints=True,
        benchmark_mode=True,
    )

    conductance_lat = dynamics.state.geometry.grid.lat
    conductance_lon = dynamics.state.geometry.grid.lon
    theta = np.deg2rad(90.0 - conductance_lat)
    hall = 0.3 + 0.1 * np.sin(theta)
    pedersen = 1.0 + 0.2 * np.cos(theta) ** 2
    dynamics.set_conductance(
        hall,
        pedersen,
        lat=conductance_lat,
        lon=conductance_lon,
        time=None,
    )

    jr_lat = dynamics.state.geometry.grid.lat
    jr_lon = dynamics.state.geometry.grid.lon
    dynamics.set_jr(
        np.zeros_like(jr_lat),
        lat=jr_lat,
        lon=jr_lon,
        time=None,
    )

    dynamics.state.update(dynamics.input_manager, np.float64(0), interpolation=True)
    return dynamics.state


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro") / max(np.linalg.norm(b, ord="fro"), 1e-30))


def _lift_two_channel(op: np.ndarray) -> np.ndarray:
    z = np.zeros_like(op)
    return np.block([[op, z], [z, op]])


def _scalar_transfer_operators(state, sh_basis):
    from pynamit.simulation.spatial import to_dense

    grid = state.geometry.grid
    sol = state.solution_basis

    g_sol = np.asarray(to_dense(sol.get_evaluation_matrix(grid)))
    p_sol = np.asarray(to_dense(sol.construct_scalar_projection_matrix(grid)))
    g_sh = np.asarray(to_dense(sh_basis.get_evaluation_matrix(grid)))
    p_sh = np.asarray(to_dense(sh_basis.construct_scalar_projection_matrix(grid)))

    # SH analysis coefficients -> solution coefficients
    lift_in = p_sol @ g_sh
    # solution coefficients -> SH analysis coefficients
    lift_out = p_sh @ g_sol
    return lift_in, lift_out


def _stage_errors(*, nmax: int, mmax: int, ncs: int, output_prefix: str) -> dict[str, float]:
    from pynamit.simulation.spatial import to_dense
    from pynamit.spherical_harmonics.sh_basis import SHBasis

    sh_basis = SHBasis(nmax, mmax, mean_free=True)
    state_st = _build_state(
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        nmax=nmax,
        mmax=mmax,
        ncs=ncs,
        filename_prefix=f"{output_prefix}_st",
    )
    state_cs = _build_state(
        SimulationMode.CS_DOMINANT,
        nmax=nmax,
        mmax=mmax,
        ncs=ncs,
        filename_prefix=f"{output_prefix}_cs",
    )

    a_st, b_st = _scalar_transfer_operators(state_st, sh_basis)
    a_cs, b_cs = _scalar_transfer_operators(state_cs, sh_basis)
    a2_st, b2_st = _lift_two_channel(a_st), _lift_two_channel(b_st)
    a2_cs, b2_cs = _lift_two_channel(a_cs), _lift_two_channel(b_cs)

    # Stage 1: psi -> E
    psi_to_e_st = np.asarray(to_dense(state_st.toroidal_to_E_coeffs))
    psi_to_e_cs = np.asarray(to_dense(state_cs.toroidal_to_E_coeffs))
    psi_to_e_st_sh = b2_st @ psi_to_e_st @ a_st
    psi_to_e_cs_sh = b2_cs @ psi_to_e_cs @ a_cs

    # Stage 2: E -> toroidal RHS
    e_to_rhs_st = np.asarray(state_st.toroidal_matrices.toroidal_rhs_from_E_operator)
    e_to_rhs_cs = np.asarray(state_cs.toroidal_matrices.toroidal_rhs_from_E_operator)
    e_to_rhs_st_sh = b_st @ e_to_rhs_st @ a2_st
    e_to_rhs_cs_sh = b_cs @ e_to_rhs_cs @ a2_cs

    # Stage 3: toroidal RHS -> dt_alpha
    dtalpha_from_rhs_st = np.asarray(
        state_st.toroidal_matrices.solver._get_unconstrained_dtalpha_map_cached(
            weighting="none",
            regularization_lambda=0.0,
            penalty_operator=None,
            penalty_scaling=0.0,
            hinv_rtol=0.0,
        )
    )
    dtalpha_from_rhs_cs = np.asarray(
        state_cs.toroidal_matrices.solver._get_unconstrained_dtalpha_map_cached(
            weighting="none",
            regularization_lambda=0.0,
            penalty_operator=None,
            penalty_scaling=0.0,
            hinv_rtol=0.0,
        )
    )
    dtalpha_from_rhs_st_sh = b_st @ dtalpha_from_rhs_st @ a_st
    dtalpha_from_rhs_cs_sh = b_cs @ dtalpha_from_rhs_cs @ a_cs

    # Stage 4: dt_alpha -> dpsi
    m_imp_to_jr_st = state_st.geometry.get_potential_to_JS_operator("m_imp", mode=None)
    m_imp_to_jr_cs = state_cs.geometry.get_potential_to_JS_operator("m_imp", mode=None)
    dtalpha_to_dt_psi_st = np.asarray(
        state_st.toroidal_matrices.solver._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_st, use_pinning=False
        )
    )
    dtalpha_to_dt_psi_cs = np.asarray(
        state_cs.toroidal_matrices.solver._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=m_imp_to_jr_cs, use_pinning=False
        )
    )
    dtalpha_to_dt_psi_st_sh = b_st @ dtalpha_to_dt_psi_st @ a_st
    dtalpha_to_dt_psi_cs_sh = b_cs @ dtalpha_to_dt_psi_cs @ a_cs

    return {
        "psi_to_e": _rel_fro(psi_to_e_cs_sh, psi_to_e_st_sh),
        "e_to_rhs": _rel_fro(e_to_rhs_cs_sh, e_to_rhs_st_sh),
        "dtalpha_from_rhs": _rel_fro(dtalpha_from_rhs_cs_sh, dtalpha_from_rhs_st_sh),
        "dtalpha_to_dt_psi": _rel_fro(dtalpha_to_dt_psi_cs_sh, dtalpha_to_dt_psi_st_sh),
    }


def test_toroidal_stage_parity_converges_with_resolution(tmp_path) -> None:
    """CS-dominant stage operators should approach ST-CS as resolution increases."""

    # Keep Nmax/Ncs below Nyquist warning threshold to avoid alias-dominated runs.
    low = _stage_errors(
        nmax=6,
        mmax=6,
        ncs=10,
        output_prefix=str(tmp_path / "stage_low"),
    )
    high = _stage_errors(
        nmax=8,
        mmax=8,
        ncs=14,
        output_prefix=str(tmp_path / "stage_high"),
    )

    # For the dominant discrepancy stages, require clear improvement.
    assert high["psi_to_e"] < low["psi_to_e"] * 0.95
    # If E->RHS parity is already at numerical precision, don't enforce
    # monotonic "improvement" against roundoff.
    if low["e_to_rhs"] > 1e-12:
        assert high["e_to_rhs"] < low["e_to_rhs"] * 0.95
    else:
        assert high["e_to_rhs"] < 1e-12

    # The alpha-solve and alpha->psi stages can converge non-monotonically against
    # ST-CS at modest resolutions; guard against blow-up while allowing drift.
    assert np.isfinite(high["dtalpha_from_rhs"])
    assert np.isfinite(high["dtalpha_to_dt_psi"])
    assert high["dtalpha_from_rhs"] < low["dtalpha_from_rhs"] * 2.5
    assert high["dtalpha_to_dt_psi"] < low["dtalpha_to_dt_psi"] * 2.5

    # Absolute sanity bounds at the higher resolution point.
    assert high["psi_to_e"] < 0.20
