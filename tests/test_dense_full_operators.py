"""Smoke tests for dense full-operator evolution paths."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pynamit.math.integration import EulerIntegrator
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, SimulationMode


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_dense_full_operators_legacy_smoke(tmp_path: Path) -> None:
    """Legacy mode should run with dense full operators enabled."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=8,
            dynamics_mode=DynamicsMode.LEGACY,
            simulation_mode=SimulationMode.PURE_SPECTRAL,
            least_squares_solver="svd",
            dense_full_operators=True,
        )

        settings = sim.io.load_dataset("settings")
        state = sim.io.load_dataset("state")
        assert settings is not None
        assert state is not None
        assert int(settings.attrs.get("dense_full_operators", 0)) == 1
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_legacy_runtime_euler_uses_reduced_gauge_system(tmp_path: Path, monkeypatch) -> None:
    """Legacy CS-dominant Euler stepping should evolve in reduced gauge coordinates."""
    from pynamit.simulation.state import State

    original_step = EulerIntegrator.step
    original_system = State.get_m_ind_reduced_system
    seen: dict[str, int | list[int]] = {}

    def recording_system(self, *args, **kwargs):
        system = original_system(self, *args, **kwargs)
        seen["n_total"] = system.n_total
        seen["n_reduced"] = system.n_reduced
        return system

    def recording_step(self, y, dt, **kwargs):
        seen.setdefault("step_sizes", []).append(int(np.asarray(y).size))
        return original_step(self, y, dt, **kwargs)

    monkeypatch.setattr(State, "get_m_ind_reduced_system", recording_system)
    monkeypatch.setattr(EulerIntegrator, "step", recording_step)

    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=4,
            Mmax=1,
            Ncs=10,
            dynamics_mode=DynamicsMode.LEGACY,
            simulation_mode=SimulationMode.CS_DOMINANT,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
            steady_state_initialization=False,
        )

        assert seen["step_sizes"]
        assert seen["n_reduced"] < seen["n_total"]
        assert seen["step_sizes"][0] == seen["n_reduced"]
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_legacy_m_imp_feedback_uses_reduced_coordinates(tmp_path: Path, monkeypatch) -> None:
    """Legacy CS-dominant m_imp feedback should solve reduced coordinates then expand."""
    from pynamit.simulation.induction.poloidal_solver import MImpFeedbackSystem

    original_expand = MImpFeedbackSystem.expand_solution
    seen: dict[str, list[int]] = {"reduced_sizes": [], "full_sizes": []}

    def recording_expand(self, m_imp_solution):
        m_imp_arr = np.asarray(m_imp_solution)
        result = original_expand(self, m_imp_solution)
        seen["reduced_sizes"].append(
            int(m_imp_arr.shape[0] if m_imp_arr.ndim > 1 else m_imp_arr.size)
        )
        result_arr = np.asarray(result)
        seen["full_sizes"].append(
            int(result_arr.shape[0] if result_arr.ndim > 1 else result_arr.size)
        )
        return result

    monkeypatch.setattr(MImpFeedbackSystem, "expand_solution", recording_expand)

    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=4,
            Mmax=1,
            Ncs=10,
            dynamics_mode=DynamicsMode.LEGACY,
            simulation_mode=SimulationMode.CS_DOMINANT,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
            steady_state_initialization=False,
        )

        assert (
            sim.state.m_imp_feedback_system.problem.solution_size
            < sim.state.solution_space.index_length
        )
        assert seen["reduced_sizes"]
        assert seen["full_sizes"]
        assert seen["reduced_sizes"][0] == sim.state.m_imp_feedback_system.problem.solution_size
        assert seen["full_sizes"][0] == sim.state.solution_space.index_length
        assert seen["reduced_sizes"][0] < seen["full_sizes"][0]
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_dense_full_operators_full_induction_smoke(tmp_path: Path) -> None:
    """Full-induction mode should run with dense coupled operator stepping."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=8,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=True,
        )

        settings = sim.io.load_dataset("settings")
        state = sim.io.load_dataset("state")
        assert settings is not None
        assert state is not None
        assert int(settings.attrs.get("dense_full_operators", 0)) == 1

        psi_name = "SH_psi" if "SH_psi" in state.data_vars else "CS_psi"
        m_ind_name = "SH_m_ind" if "SH_m_ind" in state.data_vars else "CS_m_ind"
        psi_t1 = np.asarray(state[psi_name].values[1]).reshape(-1)
        m_ind_t1 = np.asarray(state[m_ind_name].values[1]).reshape(-1)
        assert np.all(np.isfinite(psi_t1))
        assert np.all(np.isfinite(m_ind_t1))
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_coupled_sparse_dense_parity(tmp_path: Path) -> None:
    """Dense and sparse coupled Euler paths should match for full induction."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        common_kwargs = dict(
            final_time=2.0,
            dt=1.0,
            plotsteps=1,
            Nmax=8,
            Mmax=4,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
        )

        sim_sparse = run_pynamit(**common_kwargs, dense_full_operators=False)
        sim_dense = run_pynamit(**common_kwargs, dense_full_operators=True)

        ds_sparse = sim_sparse.io.load_dataset("state")
        ds_dense = sim_dense.io.load_dataset("state")
        assert ds_sparse is not None and ds_dense is not None

        psi_name = "SH_psi" if "SH_psi" in ds_sparse.data_vars else "CS_psi"
        m_ind_name = "SH_m_ind" if "SH_m_ind" in ds_sparse.data_vars else "CS_m_ind"

        psi_sparse = np.asarray(ds_sparse[psi_name].values[-1]).reshape(-1)
        psi_dense = np.asarray(ds_dense[psi_name].values[-1]).reshape(-1)
        m_sparse = np.asarray(ds_sparse[m_ind_name].values[-1]).reshape(-1)
        m_dense = np.asarray(ds_dense[m_ind_name].values[-1]).reshape(-1)

        psi_rel = np.linalg.norm(psi_sparse - psi_dense) / max(
            np.linalg.norm(psi_sparse), np.linalg.norm(psi_dense), 1e-30
        )
        m_rel = np.linalg.norm(m_sparse - m_dense) / max(
            np.linalg.norm(m_sparse), np.linalg.norm(m_dense), 1e-30
        )

        assert psi_rel < 1e-12
        assert m_rel < 1e-12
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize(
    "simulation_mode",
    [
        SimulationMode.PURE_SPECTRAL,
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        SimulationMode.CS_DOMINANT,
    ],
)
def test_full_induction_coupled_euler_integrator_object_parity(
    tmp_path: Path, simulation_mode: SimulationMode
) -> None:
    """Coupled Euler step must match direct EulerIntegrator for same L, K, y, dt."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=12,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=simulation_mode,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=True,
            connect_hemispheres=True,
            benchmark_mode=True,
        )
        state = sim.state
        n = state.solution_space.index_length
        m = 2 * n

        rng = np.random.default_rng(12345)
        y = rng.standard_normal((2, n))
        K = rng.standard_normal((2, n))
        dt = 0.2

        L = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=True))
        y_step = np.asarray(
            state._evolve_linear_state(y=y, dt=dt, linear_operator=L, forcing=K)
        ).reshape(2, n)

        y_flat = np.asarray(y).reshape(m)
        K_flat = np.asarray(K).reshape(m)
        euler = EulerIntegrator()
        y_euler = np.asarray(
            euler.step(y=y_flat, dt=dt, rates_func=lambda vec, _t: L @ np.asarray(vec) + K_flat)
        ).reshape(2, n)

        np.testing.assert_allclose(y_step, y_euler, rtol=1e-13, atol=1e-13)
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_runtime_euler_matches_reduced_gauge_system(tmp_path: Path) -> None:
    """Runtime full-induction Euler stepping should evolve in reduced gauge coordinates."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=4,
            Mmax=1,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.CS_DOMINANT,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=True,
            connect_hemispheres=True,
            northern_hemisphere_apex_constraints=True,
            benchmark_mode=True,
        )
        state = sim.state
        n = state.solution_space.index_length
        reduced_system = state.get_coupled_reduced_time_integration_system(use_dense=True)
        assert reduced_system.n_reduced < reduced_system.n_total

        rng = np.random.default_rng(20260312)
        psi = rng.standard_normal(n)
        m_ind = rng.standard_normal(n)
        dt = 0.2

        e_noind, _ = state.calculate_noind_coeffs()
        psi_step, m_ind_step = state.evolve_model_variables(
            m_ind=m_ind, dt=dt, E_coeffs_noind=e_noind, psi=psi
        )

        y0 = np.concatenate([psi, m_ind])
        z0 = reduced_system.reduce_vector(y0)
        k_reduced = reduced_system.reduce_vector(state.build_coupled_forcing(e_noind))
        a_reduced = np.asarray(reduced_system.reduced_operator.to_dense(), dtype=float)
        z1 = np.asarray(
            EulerIntegrator().step(
                y=z0, dt=dt, rates_func=lambda vec, _t: a_reduced @ np.asarray(vec) + k_reduced
            )
        )
        y1 = reduced_system.expand_vector(z1).reshape(2, n)

        np.testing.assert_allclose(np.asarray(psi_step), y1[0], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(np.asarray(m_ind_step), y1[1], rtol=1e-13, atol=1e-13)
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_exposed_coupled_matrices_and_blocks(tmp_path: Path) -> None:
    """Coupled matrix/block exposure should be shape-consistent and backend-consistent."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=8,
            Mmax=4,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
        )

        state = sim.state
        n = state.solution_space.index_length

        dense_2d = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=True))
        sparse_2d = np.asarray(state.get_coupled_induction_matrix(source="sparse", flatten=True))
        assert dense_2d.shape == (2 * n, 2 * n)
        assert sparse_2d.shape == (2 * n, 2 * n)
        np.testing.assert_allclose(dense_2d, sparse_2d, rtol=1e-12, atol=0.0)

        dense_4d = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=False))
        assert dense_4d.shape == (2, n, 2, n)
        np.testing.assert_allclose(dense_4d.reshape(2 * n, 2 * n), dense_2d, rtol=0.0, atol=0.0)

        blocks = state.get_coupled_induction_blocks(source="dense")
        required = (
            "dt_psi_from_psi",
            "dt_psi_from_m_ind",
            "dt_m_ind_from_psi",
            "dt_m_ind_from_m_ind",
        )
        for key in required:
            block = np.asarray(blocks[key])
            assert block.shape == (n, n)
        np.testing.assert_allclose(
            blocks["dt_psi_from_psi"], dense_4d[0, :, 0, :], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            blocks["dt_m_ind_from_m_ind"], dense_4d[1, :, 1, :], rtol=0.0, atol=0.0
        )
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize(
    "simulation_mode",
    [
        SimulationMode.PURE_SPECTRAL,
        SimulationMode.SPECTRAL_TRANSFORM_GL,
        SimulationMode.SPECTRAL_TRANSFORM_CS,
        SimulationMode.CS_DOMINANT,
    ],
)
def test_exposed_coupled_matrix_sparse_dense_parity_all_modes(
    tmp_path: Path, simulation_mode: SimulationMode
) -> None:
    """Dense and sparse exposed coupled matrices must match across all modes."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=12,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=simulation_mode,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            connect_hemispheres=True,
            benchmark_mode=True,
        )
        state = sim.state
        dense_2d = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=True))
        sparse_2d = np.asarray(state.get_coupled_induction_matrix(source="sparse", flatten=True))
        np.testing.assert_allclose(dense_2d, sparse_2d, rtol=0.0, atol=0.0)
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_exposed_external_forcing_matrices(tmp_path: Path) -> None:
    """Exposed dtpsi/dmind rate maps from u and jr should be internally consistent."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=8,
            Mmax=4,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
        )
        state = sim.state
        mats = state.get_external_forcing_matrices()

        required = (
            "dt_psi_from_u",
            "dt_psi_from_jr",
            "dt_m_ind_from_u",
            "dt_m_ind_from_jr",
            "dt_psi_from_E",
            "dt_m_ind_from_E",
            "E_from_u",
            "E_from_jr",
            "m_imp_from_jr",
        )
        for key in required:
            assert key in mats

        n = state.solution_space.index_length
        assert mats["dt_psi_from_u"].shape[0] == n
        assert mats["dt_psi_from_jr"].shape[0] == n
        assert mats["dt_m_ind_from_u"].shape[0] == n
        assert mats["dt_m_ind_from_jr"].shape[0] == n

        np.testing.assert_allclose(
            mats["dt_psi_from_u"], mats["dt_psi_from_E"] @ mats["E_from_u"], rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            mats["dt_psi_from_jr"], mats["dt_psi_from_E"] @ mats["E_from_jr"], rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            mats["dt_m_ind_from_u"],
            mats["dt_m_ind_from_E"] @ mats["E_from_u"],
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            mats["dt_m_ind_from_jr"],
            mats["dt_m_ind_from_E"] @ mats["E_from_jr"],
            rtol=1e-12,
            atol=0.0,
        )
        rng = np.random.default_rng(20260312)
        dt_jr = rng.standard_normal(mats["m_imp_from_jr"].shape[1])
        np.testing.assert_allclose(
            state._map_dt_jr_driver_to_dt_m_imp(dt_jr),
            mats["m_imp_from_jr"] @ dt_jr,
            rtol=1e-12,
            atol=0.0,
        )
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_toroidal_driver_balance_report_decomposes_live_forcing(tmp_path: Path) -> None:
    """Live toroidal forcing diagnostics should decompose wind and magnetic channels cleanly."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.CS_DOMINANT,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
            connect_hemispheres=True,
            wind=True,
            multi_data=True,
            benchmark_mode=True,
        )
        sim.state.update(sim.input_manager, sim.current_time, interpolation=True)
        report = sim.state.get_toroidal_driver_balance_report()
        components = report["components"]

        assert report["constraint_rows"]["ll"] > 0
        assert components["wind"]["dt_alpha_norm"] > 0.0
        assert components["magnetic_imposed"]["dt_alpha_norm"] > 0.0

        np.testing.assert_allclose(
            np.asarray(components["total_external"]["rhs_physics"]),
            np.asarray(components["wind"]["rhs_physics"])
            + np.asarray(components["Br"]["rhs_physics"])
            + np.asarray(components["magnetic_imposed"]["rhs_physics"]),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(components["total_external"]["dt_alpha"]),
            np.asarray(components["wind"]["dt_alpha"])
            + np.asarray(components["Br"]["dt_alpha"])
            + np.asarray(components["magnetic_imposed"]["dt_alpha"]),
            rtol=1e-12,
            atol=1e-12,
        )

        if "magnetic_driver" in components:
            np.testing.assert_allclose(
                np.asarray(components["residual_after_driver_subtraction"]["rhs_physics"]),
                np.asarray(components["total_external"]["rhs_physics"])
                + np.asarray(components["driver_feedback_rhs"]["rhs_physics"]),
                rtol=1e-12,
                atol=1e-12,
            )
            assert "magnetic_driver_raw" in components
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_m_imp_from_jr_matrix_is_gauge_preserving(tmp_path: Path) -> None:
    """Full-induction CS-dominant ``m_imp_from_jr`` map should preserve the m_imp gauge."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.CS_DOMINANT,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
            connect_hemispheres=True,
            benchmark_mode=True,
        )
        state = sim.state
        m_imp_from_jr = np.asarray(state.get_external_forcing_matrices()["m_imp_from_jr"])
        gauge_row = np.asarray(
            state.constraints.get_m_imp_gauge_row(state.solution_space.index_length), dtype=float
        )
        if gauge_row.size > 0:
            violation = np.linalg.norm(gauge_row @ m_imp_from_jr)
            baseline = max(np.linalg.norm(m_imp_from_jr), 1.0)
            assert violation < 1e-10 * baseline
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_hl_projection_is_lifted_from_reduced_m_imp_system(tmp_path: Path) -> None:
    """The exposed HL projector should be the full-space lift of the reduced m_imp operator."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=6,
            Mmax=3,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.CS_DOMINANT,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
            connect_hemispheres=True,
            benchmark_mode=True,
        )
        state = sim.state
        feedback_system = state.m_imp_feedback_system
        n = state.solution_space.index_length

        hl_raw = np.asarray(state.constraints.get_hl_projection_matrix(n), dtype=float)
        hl_expected = np.asarray(feedback_system.get_hl_projection_full(hl_raw), dtype=float)
        hl_exposed = np.asarray(state._get_hl_projection_matrix(n), dtype=float)
        np.testing.assert_allclose(hl_exposed, hl_expected, rtol=1e-13, atol=1e-13)

        rng = np.random.default_rng(20260312)
        values = rng.standard_normal(n)
        projected_expected = feedback_system.project_hl(values, hl_raw)
        np.testing.assert_allclose(
            np.asarray(state._project_to_hl_modes(values)),
            np.asarray(projected_expected),
            rtol=1e-13,
            atol=1e-13,
        )
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_state_exposure_apis_smoke(tmp_path: Path) -> None:
    """State exposure APIs should return self-consistent blocks/matrices."""
    previous_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        sim = run_pynamit(
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=8,
            Mmax=4,
            Ncs=10,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.SPECTRAL_TRANSFORM_CS,
            least_squares_solver="svd",
            integrator=IntegratorKind.EULER,
            dense_full_operators=False,
        )

        state = sim.state
        L_state = state.get_coupled_induction_matrix(source="dense", flatten=True)
        blocks_state = state.get_coupled_induction_blocks(source="dense")
        np.testing.assert_allclose(
            blocks_state["dt_psi_from_psi"],
            L_state[: state.solution_space.index_length, : state.solution_space.index_length],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            blocks_state["dt_m_ind_from_m_ind"],
            L_state[state.solution_space.index_length :, state.solution_space.index_length :],
            rtol=0.0,
            atol=0.0,
        )

        forcing_state = state.get_external_forcing_matrices()
        np.testing.assert_allclose(
            forcing_state["dt_psi_from_u"],
            forcing_state["dt_psi_from_E"] @ forcing_state["E_from_u"],
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            forcing_state["dt_m_ind_from_jr"],
            forcing_state["dt_m_ind_from_E"] @ forcing_state["E_from_jr"],
            rtol=1e-12,
            atol=0.0,
        )
    finally:
        os.chdir(previous_cwd)
