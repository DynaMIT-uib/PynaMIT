"""Tests for explicit postprocessing operator bundles."""

from __future__ import annotations

import numpy as np

from pynamit.math.constants import mu0
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import SimulationMode
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
        dynamics_mode="full_induction",
        simulation_mode=SimulationMode.PURE_SPECTRAL.value,
        ignore_PFAC=False,
        mainfield_kind="igrf",
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator="euler",
        least_squares_solver="svd",
    )


def _build_state():
    return _build_sim().state


def test_results_operator_bundle_matches_state_poloidal_operators() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    assert np.allclose(bundle.m_imp_to_jr, np.asarray(to_dense(state.poloidal_matrices.m_imp_to_jr)))
    assert np.allclose(bundle.m_ind_to_Br, np.asarray(to_dense(state.poloidal_matrices.m_ind_to_Br)))
    assert np.allclose(bundle.G_m_ind_to_JS, np.asarray(state.poloidal_matrices.G_m_ind_to_JS))
    assert np.allclose(bundle.G_m_imp_to_JS, np.asarray(state.poloidal_matrices.G_m_imp_to_JS))


def test_results_operator_bundle_exposes_expected_jeq_scaling() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    expected = (-state.RI / mu0) * np.asarray(to_dense(state.solution_basis.get_potential_scaling_operator()))
    assert np.allclose(bundle.m_ind_to_Jeq, expected)


def test_results_operator_bundle_builders_agree_for_live_state() -> None:
    state = _build_state()

    bundle_geometry = state.geometry.get_poloidal_results_operators()
    bundle_explicit = build_poloidal_results_operators(
        basis=state.solution_basis,
        grid=state.geometry.grid,
        RI=float(state.settings.RI),
        T_to_Ve=state.poloidal_matrices.T_to_Ve,
        RM=state.settings.RM,
    )

    for attr in ("m_ind_to_Br", "m_imp_to_jr", "m_ind_to_Jeq", "G_m_ind_to_JS", "G_m_imp_to_JS"):
        expected = np.asarray(getattr(bundle_geometry, attr))
        assert np.allclose(np.asarray(getattr(bundle_explicit, attr)), expected)


def test_results_operator_bundle_evaluation_helpers_match_explicit_application() -> None:
    state = _build_state()
    bundle = state.geometry.get_poloidal_results_operators()

    m_ind = np.arange(state.solution_basis.index_length, dtype=float)
    m_imp = np.arange(state.solution_basis.index_length, dtype=float) + 1.0

    expected_br = bundle.scalar_evaluation_matrix @ (bundle.m_ind_to_Br @ m_ind)
    expected_jr = bundle.scalar_evaluation_matrix @ (bundle.m_imp_to_jr @ m_imp)
    expected_jeq = bundle.scalar_evaluation_matrix @ (bundle.m_ind_to_Jeq @ m_ind)
    expected_js_ind = np.tensordot(bundle.G_m_ind_to_JS, m_ind, axes=([2], [0]))
    expected_js_imp = np.tensordot(bundle.G_m_imp_to_JS, m_imp, axes=([2], [0]))
    expected_js_br = np.tensordot(bundle.G_Br_to_JS, m_ind, axes=([2], [0]))

    np.testing.assert_allclose(bundle.evaluate_br(m_ind), expected_br)
    np.testing.assert_allclose(bundle.evaluate_jr(m_imp), expected_jr)
    np.testing.assert_allclose(bundle.evaluate_jeq(m_ind), expected_jeq)
    np.testing.assert_allclose(bundle.evaluate_js_from_m_ind(m_ind), expected_js_ind)
    np.testing.assert_allclose(bundle.evaluate_js_from_m_imp(m_imp), expected_js_imp)
    np.testing.assert_allclose(bundle.evaluate_js_from_br(m_ind), expected_js_br)


def test_simulation_data_results_operator_bundle_matches_live_geometry() -> None:
    sim = _build_sim()
    sim.data.pfac_matrix = np.asarray(sim.state.geometry.T_to_Ve)
    bundle_live = sim.state.geometry.get_poloidal_results_operators()
    bundle_saved = sim.data.get_poloidal_results_operators(grid=sim.state.geometry.grid)

    for attr in (
        "m_ind_to_Br",
        "m_imp_to_jr",
        "m_ind_to_Jeq",
        "G_m_ind_to_JS",
        "G_m_imp_to_JS",
        "G_Br_to_JS",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(bundle_saved, attr)),
            np.asarray(getattr(bundle_live, attr)),
        )
