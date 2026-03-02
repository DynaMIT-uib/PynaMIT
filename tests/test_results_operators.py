"""Tests for explicit postprocessing operator bundles."""

from __future__ import annotations

import numpy as np

from pynamit.math.constants import mu0
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import SimulationMode
from pynamit.simulation.spatial import to_dense
from pynamit.visualization.results_operators import (
    build_poloidal_results_operators_from_settings,
)


def _build_state():
    sim = run_pynamit(
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
    return sim.state


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
    bundle_settings = build_poloidal_results_operators_from_settings(
        state.settings,
        basis=state.solution_basis,
        grid=state.geometry.grid,
        T_to_Ve=state.poloidal_matrices.T_to_Ve,
    )

    for attr in ("m_ind_to_Br", "m_imp_to_jr", "m_ind_to_Jeq", "G_m_ind_to_JS", "G_m_imp_to_JS"):
        expected = np.asarray(getattr(bundle_geometry, attr))
        assert np.allclose(np.asarray(getattr(bundle_settings, attr)), expected)
