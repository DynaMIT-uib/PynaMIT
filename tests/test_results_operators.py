"""Tests for explicit postprocessing operator bundles."""

from __future__ import annotations

import numpy as np

from pynamit.math.constants import mu0
from pynamit.primitives.basis import get_repo_df_helmholtz_sign
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, MainfieldKind, SimulationMode
from pynamit.simulation.spatial import to_dense
from pynamit.postprocess.results_operators import (
    build_poloidal_results_operators,
)


def _build_sim():
    return run_pynamit(
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


def _build_state():
    return _build_sim().state


def test_results_operator_bundle_matches_state_poloidal_operators() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    assert np.allclose(bundle.m_imp_to_jr, np.asarray(to_dense(state.poloidal_matrices.m_imp_to_jr)))
    assert np.allclose(bundle.m_ind_to_Br, np.asarray(to_dense(state.poloidal_matrices.m_ind_to_Br)))
    assert np.allclose(
        bundle.G_m_ind_to_Jeq_vector,
        np.asarray(state.geometry.get_poloidal_results_operators().G_m_ind_to_Jeq_vector),
    )
    assert np.allclose(bundle.G_m_ind_to_JS, np.asarray(state.poloidal_matrices.G_m_ind_to_JS))
    assert np.allclose(bundle.G_m_imp_to_JS, np.asarray(state.poloidal_matrices.G_m_imp_to_JS))


def test_m_ind_to_br_pseudoinverse_is_cached_and_matches_direct_pinv() -> None:
    state = _build_state()

    pinv_cached_1 = state.poloidal_matrices.m_ind_to_Br_pinv
    pinv_cached_2 = state.poloidal_matrices.m_ind_to_Br_pinv

    m_ind_to_br = np.asarray(to_dense(state.poloidal_matrices.m_ind_to_Br))
    rcond = max(float(np.finfo(float).eps * max(m_ind_to_br.shape)), 1e-15)
    pinv_direct = np.linalg.pinv(m_ind_to_br, rcond=rcond)

    assert pinv_cached_1 is pinv_cached_2
    np.testing.assert_allclose(pinv_cached_1, pinv_direct)


def test_results_operator_bundle_exposes_expected_jeq_scaling() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    expected = (-state.RI / mu0) * np.asarray(to_dense(state.solution_space.get_potential_scaling_operator()))
    assert np.allclose(bundle.m_ind_to_Jeq, expected)


def test_sh_induced_m_ind_matches_paper_kl_scalings() -> None:
    """SH ``m_ind`` should match the paper's induced ``k,l`` coefficient family."""
    state = _build_state()
    basis = state.solution_space
    n = np.asarray(basis.n, dtype=float).reshape(-1)

    m_ind_to_br = np.asarray(to_dense(state.poloidal_matrices.m_ind_to_Br), dtype=float)
    m_ind_to_jeq = np.asarray(state.geometry.get_poloidal_results_operators().m_ind_to_Jeq, dtype=float)

    expected_br = np.diag(n * (n + 1.0))
    expected_jeq = np.diag((-(state.RI / mu0)) * (2.0 * n + 1.0))

    np.testing.assert_allclose(m_ind_to_br, expected_br, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(m_ind_to_jeq, expected_jeq, rtol=1e-12, atol=1e-12)


def test_results_operator_bundle_builders_agree_for_live_state() -> None:
    state = _build_state()

    bundle_geometry = state.geometry.get_poloidal_results_operators()
    bundle_explicit = build_poloidal_results_operators(
        basis=state.solution_space,
        grid=state.geometry.grid,
        RI=float(state.settings.RI),
        T_to_Ve=state.poloidal_matrices.T_to_Ve,
        RM=state.settings.RM,
    )

    for attr in (
        "m_ind_to_Br",
        "m_imp_to_jr",
        "m_ind_to_Jeq",
        "G_m_ind_to_Jeq_vector",
        "G_m_ind_to_JS",
        "G_m_imp_to_JS",
    ):
        expected = np.asarray(getattr(bundle_geometry, attr))
        assert np.allclose(np.asarray(getattr(bundle_explicit, attr)), expected)


def test_results_operator_bundle_is_cached_on_geometry() -> None:
    state = _build_state()
    bundle_1 = state.geometry.get_poloidal_results_operators()
    bundle_2 = state.geometry.get_poloidal_results_operators()

    assert bundle_1 is bundle_2


def test_results_operator_bundle_evaluation_helpers_match_explicit_application() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    m_ind = np.arange(state.solution_space.index_length, dtype=float)
    m_imp = np.arange(state.solution_space.index_length, dtype=float) + 1.0

    expected_br = bundle.scalar_evaluation_matrix @ (bundle.m_ind_to_Br @ m_ind)
    expected_jr = bundle.scalar_evaluation_matrix @ (bundle.m_imp_to_jr @ m_imp)
    expected_jeq = bundle.scalar_evaluation_matrix @ (bundle.m_ind_to_Jeq @ m_ind)
    expected_jeq_vector = np.tensordot(bundle.G_m_ind_to_Jeq_vector, m_ind, axes=([2], [0]))
    expected_js_ind = np.tensordot(bundle.G_m_ind_to_JS, m_ind, axes=([2], [0]))
    expected_js_imp = np.tensordot(bundle.G_m_imp_to_JS, m_imp, axes=([2], [0]))
    expected_js_br = np.tensordot(bundle.G_Br_to_JS, m_ind, axes=([2], [0]))
    expected_js_total = expected_js_ind + expected_js_imp + expected_js_br

    np.testing.assert_allclose(bundle.evaluate_br(m_ind), expected_br)
    np.testing.assert_allclose(bundle.evaluate_jr(m_imp), expected_jr)
    np.testing.assert_allclose(bundle.evaluate_jeq(m_ind), expected_jeq)
    np.testing.assert_allclose(bundle.evaluate_jeq_vector(m_ind), expected_jeq_vector)
    np.testing.assert_allclose(bundle.evaluate_js_from_m_ind(m_ind), expected_js_ind)
    np.testing.assert_allclose(bundle.evaluate_js_from_m_imp(m_imp), expected_js_imp)
    np.testing.assert_allclose(bundle.evaluate_js_from_br(m_ind), expected_js_br)
    np.testing.assert_allclose(
        bundle.evaluate_runtime_js(m_ind=m_ind, m_imp=m_imp, br_coeffs=m_ind),
        expected_js_total,
    )


def test_jeq_vector_matches_curl_of_conventional_equivalent_current_function() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    m_ind = np.arange(state.solution_space.index_length, dtype=float) + 0.5
    psi_eq = np.asarray(bundle.m_ind_to_Jeq) @ m_ind
    curl = np.asarray(
        to_dense(state.solution_space.get_curl_matrix(state.geometry.grid)),
        dtype=float,
    ).reshape(2, -1, state.solution_space.index_length)
    expected = (1.0 / (float(state.RI) * float(get_repo_df_helmholtz_sign()))) * np.tensordot(
        curl, psi_eq, axes=([2], [0])
    )

    np.testing.assert_allclose(bundle.evaluate_jeq_vector(m_ind), expected)


def test_simulation_data_results_operator_bundle_matches_live_geometry() -> None:
    sim = _build_sim()
    sim.data.pfac_matrix = np.asarray(sim.state.geometry.T_to_Ve)
    bundle_live = sim.state.geometry.get_poloidal_results_operators()
    bundle_saved = sim.data.get_poloidal_results_operators(grid=sim.state.geometry.grid)

    for attr in (
        "m_ind_to_Br",
        "m_imp_to_jr",
        "m_ind_to_Jeq",
        "G_m_ind_to_Jeq_vector",
        "G_m_ind_to_JS",
        "G_m_imp_to_JS",
        "G_Br_to_JS",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(bundle_saved, attr)),
            np.asarray(getattr(bundle_live, attr)),
        )
