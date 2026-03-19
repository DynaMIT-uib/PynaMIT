"""Focused sign checks for induced poloidal operators."""

from __future__ import annotations

import numpy as np

import pynamit.primitives.basis as basis_mod
from pynamit.math.constants import mu0
from pynamit.primitives.basis import get_repo_df_helmholtz_sign
from pynamit.postprocess.ground_response import build_ground_magnetic_response_operators
from pynamit.primitives.grid import Grid
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, MainfieldKind, SimulationMode
from pynamit.simulation.spatial.geometry_utils import to_dense


def _build_state():
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=8,
        Mmax=4,
        Ncs=10,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator=IntegratorKind.EULER,
        least_squares_solver="svd",
    )
    return sim.state


def _build_legacy_state():
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=8,
        Mmax=4,
        Ncs=10,
        dynamics_mode=DynamicsMode.LEGACY,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator=IntegratorKind.EULER,
        least_squares_solver="svd",
    )
    return sim.state


def test_e_df_to_dm_ind_dt_matches_closed_form_sh_sign() -> None:
    """Faraday should track the active repo df-coefficient convention in SH."""
    state = _build_state()
    basis = state.solution_space
    degrees = np.asarray(basis.n, dtype=float).reshape(-1)
    nonzero = degrees > 0.0

    expected = (
        float(get_repo_df_helmholtz_sign())
        * np.ones(np.count_nonzero(nonzero), dtype=float)
        / float(state.RI)
    )
    derived = np.full(np.count_nonzero(nonzero), state.poloidal_matrices.E_df_to_d_m_ind_dt)

    np.testing.assert_allclose(derived, expected, rtol=0.0, atol=1e-18)
    assert state.poloidal_matrices.E_df_to_d_m_ind_dt == expected[0]


def test_pure_sh_coupled_operator_is_dissipative() -> None:
    """Pure-SH full-induction coupled operator should have no growing modes."""
    state = _build_state()
    report = state.get_coupled_stability_report(source="dense")
    assert float(report["positive_real_count"]) == 0.0
    assert float(report["max_real"]) <= 1e-10


def test_legacy_scalar_rate_operator_is_dissipative() -> None:
    """Legacy pure-SH scalar induction should have non-positive real spectrum."""
    state = _build_legacy_state()
    E_coeffs_noind, _ = state.calculate_noind_coeffs()
    full_linear_operator, _ = state.induction.build_legacy_scalar_rate_problem(E_coeffs_noind)
    evals = np.linalg.eigvals(np.asarray(full_linear_operator, dtype=float)).real

    assert float(np.max(evals)) <= 1e-10


def test_induced_js_matches_magnetic_jump_current_without_rm() -> None:
    """For ``RM=None``, induced runtime ``JS`` should equal the magnetic jump current."""
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()
    m_ind = np.arange(state.solution_space.index_length, dtype=float) + 0.5

    scaling = np.asarray(
        to_dense(state.solution_space.get_potential_scaling_operator()),
        dtype=float,
    )
    curl = np.asarray(
        to_dense(state.solution_space.get_curl_matrix(state.geometry.grid)),
        dtype=float,
    ).reshape(2, -1, state.solution_space.index_length)
    expected = (-1.0 / (float(mu0) * float(get_repo_df_helmholtz_sign()))) * np.tensordot(
        curl, scaling @ m_ind, axes=([2], [0])
    )

    np.testing.assert_allclose(
        bundle.evaluate_js_from_m_ind(m_ind),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_geometry_jump_current_helpers_match_runtime_operators() -> None:
    """Geometry helper operators should match the live poloidal runtime operators."""
    state = _build_state()

    np.testing.assert_allclose(
        np.asarray(state.geometry.G_Ve_to_JS_closure, dtype=float),
        np.asarray(state.poloidal_matrices.G_Ve_to_JS_closure, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state.geometry.G_Ve_to_JS, dtype=float),
        np.asarray(state.poloidal_matrices.G_Ve_to_JS, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state.geometry.T_to_Ve, dtype=float),
        np.asarray(state.poloidal_matrices.T_to_Ve, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_df_sign_flip_preserves_magnetic_maps_but_flips_df_current_operators(
    monkeypatch,
) -> None:
    """A df-basis sign change should not alter physical magnetic/grid observables."""
    state = _build_state()
    grid = Grid(lat=np.array([60.0]), lon=np.array([10.0]))
    ground_0 = build_ground_magnetic_response_operators(
        state_spec=state.solution_space,
        ground_grid=grid,
        ionosphere_radius=float(state.RI),
    )

    m_ind_to_br_0 = np.asarray(to_dense(state.poloidal_matrices.m_ind_to_Br), dtype=float)
    m_imp_to_jr_0 = np.asarray(to_dense(state.poloidal_matrices.m_imp_to_jr), dtype=float)
    jr_to_psi_0 = np.asarray(state.toroidal_matrices.jr_to_psi_coeff_operator, dtype=float)
    g_ve_0 = np.asarray(state.poloidal_matrices.G_Ve_to_JS, dtype=float)
    g_mind_0 = np.asarray(state.poloidal_matrices.G_m_ind_to_JS, dtype=float)
    jeq_vec_0 = np.asarray(state.geometry.get_poloidal_results_operators().G_m_ind_to_Jeq_vector)
    jeq_scalar_0 = np.asarray(state.geometry.get_poloidal_results_operators().m_ind_to_Jeq)

    monkeypatch.setattr(basis_mod, "REPO_DF_HELMHOLTZ_SIGN", +1.0)
    state_flipped = _build_state()
    ground_1 = build_ground_magnetic_response_operators(
        state_spec=state_flipped.solution_space,
        ground_grid=grid,
        ionosphere_radius=float(state_flipped.RI),
    )

    m_ind_to_br_1 = np.asarray(to_dense(state_flipped.poloidal_matrices.m_ind_to_Br), dtype=float)
    m_imp_to_jr_1 = np.asarray(to_dense(state_flipped.poloidal_matrices.m_imp_to_jr), dtype=float)
    jr_to_psi_1 = np.asarray(state_flipped.toroidal_matrices.jr_to_psi_coeff_operator, dtype=float)
    g_ve_1 = np.asarray(state_flipped.poloidal_matrices.G_Ve_to_JS, dtype=float)
    g_mind_1 = np.asarray(state_flipped.poloidal_matrices.G_m_ind_to_JS, dtype=float)
    jeq_vec_1 = np.asarray(state_flipped.geometry.get_poloidal_results_operators().G_m_ind_to_Jeq_vector)
    jeq_scalar_1 = np.asarray(state_flipped.geometry.get_poloidal_results_operators().m_ind_to_Jeq)

    np.testing.assert_allclose(m_ind_to_br_1, m_ind_to_br_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m_imp_to_jr_1, m_imp_to_jr_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jr_to_psi_1, jr_to_psi_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jeq_scalar_1, jeq_scalar_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jeq_vec_1, jeq_vec_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(ground_1.radial_matrix, dtype=float),
        np.asarray(ground_0.radial_matrix, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(ground_1.horizontal_matrix, dtype=float),
        np.asarray(ground_0.horizontal_matrix, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(g_ve_1, g_ve_0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_mind_1, g_mind_0, rtol=1e-12, atol=1e-12)


def test_cf_sign_flip_preserves_physical_magnetic_and_grid_current_maps(monkeypatch) -> None:
    """Changing the internal cf sign should not alter physical grid operators."""
    state = _build_state()
    grid = Grid(lat=np.array([60.0]), lon=np.array([10.0]))
    ground_0 = build_ground_magnetic_response_operators(
        state_spec=state.solution_space,
        ground_grid=grid,
        ionosphere_radius=float(state.RI),
    )
    bundle_0 = state.geometry.get_poloidal_results_operators()

    m_ind_to_br_0 = np.asarray(to_dense(state.poloidal_matrices.m_ind_to_Br), dtype=float)
    m_imp_to_jr_0 = np.asarray(to_dense(state.poloidal_matrices.m_imp_to_jr), dtype=float)
    jr_to_psi_0 = np.asarray(state.toroidal_matrices.jr_to_psi_coeff_operator, dtype=float)
    g_btor_0 = np.asarray(bundle_0.G_B_tor_to_JS, dtype=float)
    g_mimp_0 = np.asarray(bundle_0.G_m_imp_to_JS, dtype=float)

    monkeypatch.setattr(basis_mod, "REPO_CF_HELMHOLTZ_SIGN", +1.0)
    state_flipped = _build_state()
    ground_1 = build_ground_magnetic_response_operators(
        state_spec=state_flipped.solution_space,
        ground_grid=grid,
        ionosphere_radius=float(state_flipped.RI),
    )
    bundle_1 = state_flipped.geometry.get_poloidal_results_operators()

    np.testing.assert_allclose(
        np.asarray(to_dense(state_flipped.poloidal_matrices.m_ind_to_Br), dtype=float),
        m_ind_to_br_0,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(to_dense(state_flipped.poloidal_matrices.m_imp_to_jr), dtype=float),
        m_imp_to_jr_0,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(state_flipped.toroidal_matrices.jr_to_psi_coeff_operator, dtype=float),
        jr_to_psi_0,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle_1.G_B_tor_to_JS, dtype=float), g_btor_0, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(bundle_1.G_m_imp_to_JS, dtype=float), g_mimp_0, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ground_1.radial_matrix, dtype=float),
        np.asarray(ground_0.radial_matrix, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(ground_1.horizontal_matrix, dtype=float),
        np.asarray(ground_0.horizontal_matrix, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def _build_hc_wind_dynamics(*, enable_fast_input_path: bool = False):
    """Build the legacy HC+wind configuration used for end-to-end sign checks."""
    return run_pynamit(
        final_time=0.1,
        dt=1e-2,
        Nmax=10,
        Mmax=8,
        Ncs=18,
        mainfield_kind=MainfieldKind.IGRF,
        ignore_PFAC=False,
        connect_hemispheres=True,
        latitude_boundary=50,
        wind=True,
        steady_state_initialization=False,
        benchmark_mode=True,
        enable_fast_input_path=enable_fast_input_path,
    )


def _get_state_coeff_array(dynamics) -> np.ndarray:
    """Return the concatenated final ``[m_ind, m_imp]`` coefficient vector."""
    return np.hstack(
        (
            dynamics.output_timeseries.datasets["state"]["SH_m_ind"].values[-1],
            dynamics.output_timeseries.datasets["state"]["SH_m_imp"].values[-1],
        )
    )


def test_df_sign_flip_preserves_hc_wind_solution_and_fast_input_path(monkeypatch) -> None:
    """HC wind runs should stay physically invariant under an internal df-sign flip."""
    baseline = _build_hc_wind_dynamics(enable_fast_input_path=False)
    baseline_coeffs = _get_state_coeff_array(baseline)

    monkeypatch.setattr(basis_mod, "REPO_DF_HELMHOLTZ_SIGN", +1.0)

    flipped_reference = _build_hc_wind_dynamics(enable_fast_input_path=False)
    flipped_fast = _build_hc_wind_dynamics(enable_fast_input_path=True)

    flipped_reference_coeffs = _get_state_coeff_array(flipped_reference)
    flipped_fast_coeffs = _get_state_coeff_array(flipped_fast)

    np.testing.assert_allclose(
        flipped_reference_coeffs, baseline_coeffs, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        flipped_fast_coeffs, flipped_reference_coeffs, rtol=1e-9, atol=1e-12
    )


def test_cf_sign_flip_preserves_hc_scaled_induction_feedback_matrix(monkeypatch) -> None:
    """The HC legacy induction feedback matrix should be cf-sign invariant."""
    state = _build_hc_wind_dynamics(enable_fast_input_path=False).state
    base_scale = state.poloidal_matrices.E_df_to_d_m_ind_dt
    base_matrix = np.asarray(state.m_ind_to_E_df_matrix, dtype=float)

    monkeypatch.setattr(basis_mod, "REPO_CF_HELMHOLTZ_SIGN", +1.0)
    flipped = _build_hc_wind_dynamics(enable_fast_input_path=False).state
    flipped_scale = flipped.poloidal_matrices.E_df_to_d_m_ind_dt
    flipped_matrix = np.asarray(flipped.m_ind_to_E_df_matrix, dtype=float)

    np.testing.assert_allclose(
        flipped_scale * flipped_matrix,
        base_scale * base_matrix,
        rtol=1e-11,
        atol=1e-12,
    )


def test_legacy_steady_state_uses_scaled_rate_problem() -> None:
    """Legacy steady state should satisfy the same scaled rate equation as runtime evolution."""
    state = _build_legacy_state()
    E_coeffs_noind, _ = state.calculate_noind_coeffs()
    _, steady_state_m_ind = state.solve_steady_state_model_variables(
        E_coeffs_noind, update_state=False
    )

    full_linear_operator, full_forcing = state.induction.build_legacy_scalar_rate_problem(
        E_coeffs_noind
    )
    reduced_system = state.get_m_ind_reduced_system(linear_operator=full_linear_operator)
    assert reduced_system.reduced_operator is not None

    steady_state_reduced = reduced_system.reduce_vector(np.asarray(steady_state_m_ind, dtype=float))
    forcing_reduced = reduced_system.reduce_vector(np.asarray(full_forcing, dtype=float))
    residual = np.linalg.norm(
        reduced_system.reduced_operator.matvec(steady_state_reduced) + forcing_reduced
    ) / max(np.linalg.norm(forcing_reduced), 1e-30)

    assert residual < 1e-10
