"""Smoke tests for dense full-operator evolution paths."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pynamit.math.integration import EulerIntegrator
from pynamit.simulation.runner import run_pynamit


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
            dynamics_mode="legacy",
            simulation_mode="pure_spectral",
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
            dynamics_mode="full_induction",
            simulation_mode="spectral_transform_cs",
            least_squares_solver="svd",
            integrator="euler",
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
            dynamics_mode="full_induction",
            simulation_mode="spectral_transform_cs",
            least_squares_solver="svd",
            integrator="euler",
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
        "pure_spectral",
        "spectral_transform_gl",
        "spectral_transform_cs",
        "cs_dominant",
    ],
)
def test_full_induction_coupled_euler_integrator_object_parity(
    tmp_path: Path,
    simulation_mode: str,
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
            dynamics_mode="full_induction",
            simulation_mode=simulation_mode,
            least_squares_solver="svd",
            integrator="euler",
            dense_full_operators=True,
            connect_hemispheres=True,
            benchmark_mode=True,
        )
        state = sim.state
        n = state.solution_basis.index_length
        m = 2 * n

        rng = np.random.default_rng(12345)
        y = rng.standard_normal((2, n))
        K = rng.standard_normal((2, n))
        dt = 0.2

        L = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=True))
        y_step = np.asarray(
            state._evolve_linear_state(
                y=y,
                dt=dt,
                linear_operator=L,
                forcing=K,
            )
        ).reshape(2, n)

        y_flat = np.asarray(y).reshape(m)
        K_flat = np.asarray(K).reshape(m)
        euler = EulerIntegrator()
        y_euler = np.asarray(
            euler.step(
                y=y_flat,
                dt=dt,
                rates_func=lambda vec, _t: L @ np.asarray(vec) + K_flat,
            )
        ).reshape(2, n)

        np.testing.assert_allclose(y_step, y_euler, rtol=1e-13, atol=1e-13)
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
            dynamics_mode="full_induction",
            simulation_mode="spectral_transform_cs",
            least_squares_solver="svd",
            integrator="euler",
            dense_full_operators=False,
        )

        state = sim.state
        n = state.solution_basis.index_length

        dense_2d = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=True))
        sparse_2d = np.asarray(state.get_coupled_induction_matrix(source="sparse", flatten=True))
        assert dense_2d.shape == (2 * n, 2 * n)
        assert sparse_2d.shape == (2 * n, 2 * n)
        np.testing.assert_allclose(dense_2d, sparse_2d, rtol=1e-12, atol=0.0)

        dense_4d = np.asarray(state.get_coupled_induction_matrix(source="dense", flatten=False))
        assert dense_4d.shape == (2, n, 2, n)
        np.testing.assert_allclose(dense_4d.reshape(2 * n, 2 * n), dense_2d, rtol=0.0, atol=0.0)

        blocks = state.get_coupled_induction_blocks(source="dense")
        required = ("dt_psi_from_psi", "dt_psi_from_m_ind", "dt_m_ind_from_psi", "dt_m_ind_from_m_ind")
        for key in required:
            block = np.asarray(blocks[key])
            assert block.shape == (n, n)
        np.testing.assert_allclose(blocks["dt_psi_from_psi"], dense_4d[0, :, 0, :], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(blocks["dt_m_ind_from_m_ind"], dense_4d[1, :, 1, :], rtol=0.0, atol=0.0)
    finally:
        os.chdir(previous_cwd)


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize(
    "simulation_mode",
    [
        "pure_spectral",
        "spectral_transform_gl",
        "spectral_transform_cs",
        "cs_dominant",
    ],
)
def test_exposed_coupled_matrix_sparse_dense_parity_all_modes(
    tmp_path: Path,
    simulation_mode: str,
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
            dynamics_mode="full_induction",
            simulation_mode=simulation_mode,
            least_squares_solver="svd",
            integrator="euler",
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
            dynamics_mode="full_induction",
            simulation_mode="spectral_transform_cs",
            least_squares_solver="svd",
            integrator="euler",
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

        n = state.solution_basis.index_length
        assert mats["dt_psi_from_u"].shape[0] == n
        assert mats["dt_psi_from_jr"].shape[0] == n
        assert mats["dt_m_ind_from_u"].shape[0] == n
        assert mats["dt_m_ind_from_jr"].shape[0] == n

        np.testing.assert_allclose(
            mats["dt_psi_from_u"],
            mats["dt_psi_from_E"] @ mats["E_from_u"],
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            mats["dt_psi_from_jr"],
            mats["dt_psi_from_E"] @ mats["E_from_jr"],
            rtol=1e-12,
            atol=0.0,
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
            dynamics_mode="full_induction",
            simulation_mode="spectral_transform_cs",
            least_squares_solver="svd",
            integrator="euler",
            dense_full_operators=False,
        )

        state = sim.state
        L_state = state.get_coupled_induction_matrix(source="dense", flatten=True)
        blocks_state = state.get_coupled_induction_blocks(source="dense")
        np.testing.assert_allclose(
            blocks_state["dt_psi_from_psi"],
            L_state[: state.solution_basis.index_length, : state.solution_basis.index_length],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            blocks_state["dt_m_ind_from_m_ind"],
            L_state[state.solution_basis.index_length :, state.solution_basis.index_length :],
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
