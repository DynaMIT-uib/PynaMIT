"""Regression tests for explicit exponential solver mode combinations."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.math.integration import ExponentialIntegrator
from pynamit.utils import JAX_AVAILABLE
from pynamit.simulation.settings import (
    DynamicsMode,
    ExponentialSolverKind,
    IntegratorKind,
    MainfieldKind,
    SimulationMode,
)


def _run_full_induction_exponential_mode(
    tmp_path, *, dense_full_operators: bool, exponential_solver: str
):
    from pynamit.simulation.runner import run_pynamit

    sim = run_pynamit(
        run_directory=str(
            tmp_path / f"exp_mode_{exponential_solver}_{'dense' if dense_full_operators else 'mf'}"
        ),
        final_time=1.0,
        dt=1.0,
        plotsteps=1,
        Nmax=5,
        Mmax=2,
        Ncs=6,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=False,
        connect_hemispheres=True,
        least_squares_solver="svd",
        integrator=IntegratorKind.EXPONENTIAL,
        dense_full_operators=dense_full_operators,
        exponential_solver=exponential_solver,
    )

    ds = sim.io.load_dataset("state")
    steady_ds = sim.io.load_dataset("steady_state")
    psi = ds["SH_psi"].values[-1]
    m_ind = ds["SH_m_ind"].values[-1]
    psi_ss = steady_ds["SH_psi"].values[-1]
    m_ind_ss = steady_ds["SH_m_ind"].values[-1]
    return (
        float(np.linalg.norm(psi)),
        float(np.linalg.norm(m_ind)),
        float(np.linalg.norm(psi_ss)),
        float(np.linalg.norm(m_ind_ss)),
    )


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize(
    "label,dense_full_operators,exponential_solver,expected_psi_norm,expected_mind_norm",
    [
        ("dense_expm", True, "expm", 5.048226712788322e-10, 1.6874470337361025e-09),
        (
            "dense_expm_multiply",
            True,
            "expm_multiply",
            5.048226712788322e-10,
            1.6874470337361025e-09,
        ),
        (
            "matrixfree_expm_multiply",
            False,
            "expm_multiply",
            5.048226712788322e-10,
            1.6874470337361025e-09,
        ),
    ],
)
def test_full_induction_exponential_solver_modes(
    tmp_path,
    data_source,
    label,
    dense_full_operators,
    exponential_solver,
    expected_psi_norm,
    expected_mind_norm,
):
    psi_norm, mind_norm, psi_ss_norm, mind_ss_norm = _run_full_induction_exponential_mode(
        tmp_path, dense_full_operators=dense_full_operators, exponential_solver=exponential_solver
    )

    assert psi_norm == pytest.approx(expected_psi_norm, rel=1e-10, abs=0.0)
    assert mind_norm == pytest.approx(expected_mind_norm, rel=1e-10, abs=0.0)
    assert psi_ss_norm > 0.0
    assert mind_ss_norm > 0.0


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_exponential_solver_requires_dense_operators_for_expm(tmp_path):
    from pynamit.simulation.runner import run_pynamit

    with pytest.raises(
        ValueError, match="full_induction'.*exponential_solver='expm'.*dense_full_operators=True"
    ):
        run_pynamit(
            run_directory=str(tmp_path / "exp_invalid_combo"),
            final_time=0.0,
            dt=1.0,
            plotsteps=1,
            Nmax=3,
            Mmax=1,
            Ncs=4,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.PURE_SPECTRAL,
            least_squares_solver="svd",
            integrator=IntegratorKind.EXPONENTIAL,
            dense_full_operators=False,
            exponential_solver=ExponentialSolverKind.EXPM,
        )


def test_full_induction_exponential_uses_affine_forcing_step(tmp_path, monkeypatch):
    from pynamit.simulation.runner import run_pynamit

    original_step = ExponentialIntegrator.step
    seen_calls = []

    def recording_step(self, y, dt, **kwargs):
        seen_calls.append(
            {
                "forcing_is_none": kwargs.get("forcing") is None,
                "has_steady_state": kwargs.get("steady_state") is not None,
            }
        )
        return original_step(self, y, dt, **kwargs)

    monkeypatch.setattr(ExponentialIntegrator, "step", recording_step)

    run_pynamit(
        run_directory=str(tmp_path / "exp_affine_full_induction"),
        final_time=1.0,
        dt=1.0,
        plotsteps=1,
        Nmax=5,
        Mmax=2,
        Ncs=6,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=False,
        connect_hemispheres=True,
        least_squares_solver="svd",
        integrator=IntegratorKind.EXPONENTIAL,
        dense_full_operators=True,
        exponential_solver=ExponentialSolverKind.EXPM,
    )

    assert seen_calls
    assert seen_calls[0]["forcing_is_none"] is False
    assert seen_calls[0]["has_steady_state"] is False


def test_full_induction_exponential_uses_reduced_gauge_coordinates(tmp_path, monkeypatch):
    from pynamit.simulation.runner import run_pynamit
    from pynamit.simulation.state import State

    original_step = ExponentialIntegrator.step
    original_system = State.get_coupled_reduced_time_integration_system
    seen: dict[str, int | list[int]] = {}

    def recording_system(self, *args, **kwargs):
        system = original_system(self, *args, **kwargs)
        seen["n_total"] = system.n_total
        seen["n_reduced"] = system.n_reduced
        return system

    def recording_step(self, y, dt, **kwargs):
        seen.setdefault("step_sizes", []).append(int(np.asarray(y).size))
        return original_step(self, y, dt, **kwargs)

    monkeypatch.setattr(State, "get_coupled_reduced_time_integration_system", recording_system)
    monkeypatch.setattr(ExponentialIntegrator, "step", recording_step)

    run_pynamit(
        run_directory=str(tmp_path / "exp_reduced_full_induction"),
        final_time=1.0,
        dt=1.0,
        plotsteps=1,
        Nmax=4,
        Mmax=1,
        Ncs=10,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=False,
        connect_hemispheres=True,
        northern_hemisphere_apex_constraints=True,
        least_squares_solver="svd",
        integrator=IntegratorKind.EXPONENTIAL,
        dense_full_operators=True,
        exponential_solver=ExponentialSolverKind.EXPM,
    )

    assert seen["step_sizes"]
    assert seen["n_reduced"] < seen["n_total"]
    assert seen["step_sizes"][0] == seen["n_reduced"]


def test_full_induction_null_diagnostics_are_wired(tmp_path, monkeypatch):
    from pynamit.simulation.runner import run_pynamit
    from pynamit.simulation.state import State

    original_update = State._update_coupled_null_basis
    original_check = State._check_forcing_null_projection
    seen = {"update": 0, "check": 0}

    def recording_update(self, L_flat):
        seen["update"] += 1
        return original_update(self, L_flat)

    def recording_check(self, K_flat):
        seen["check"] += 1
        return original_check(self, K_flat)

    monkeypatch.setattr(State, "_update_coupled_null_basis", recording_update)
    monkeypatch.setattr(State, "_check_forcing_null_projection", recording_check)

    run_pynamit(
        run_directory=str(tmp_path / "exp_null_diagnostics"),
        final_time=1.0,
        dt=1.0,
        plotsteps=1,
        Nmax=5,
        Mmax=2,
        Ncs=6,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=False,
        connect_hemispheres=True,
        least_squares_solver="svd",
        integrator=IntegratorKind.EXPONENTIAL,
        dense_full_operators=True,
        exponential_solver=ExponentialSolverKind.EXPM,
        induction_null_diagnostics=True,
        induction_null_warn_ratio=1.0,
    )

    assert seen["update"] >= 1
    assert seen["check"] >= 1


def test_coupled_null_basis_reuses_cached_signature(monkeypatch):
    from pynamit.simulation.state import State

    state = State.__new__(State)
    state.induction_null_diagnostics = True
    state.induction_null_svd_rtol = 1e-8
    state._coupled_null_basis = None
    state._coupled_null_threshold = None
    state._coupled_null_signature = None
    state._coupled_null_warned = False

    original_svd = np.linalg.svd
    seen = {"svd": 0}

    def recording_svd(*args, **kwargs):
        seen["svd"] += 1
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)

    operator = np.eye(3, dtype=float)
    state._update_coupled_null_basis(operator)
    state._update_coupled_null_basis(operator.copy())
    operator[0, 0] = 2.0
    state._update_coupled_null_basis(operator)

    assert seen["svd"] == 2


def test_coupled_null_projection_warns_once(caplog):
    from pynamit.simulation.state import State

    state = State.__new__(State)
    state.induction_null_diagnostics = True
    state.induction_null_warn_ratio = 0.5
    state._coupled_null_basis = np.array([[1.0], [0.0]])
    state._coupled_null_warned = False

    with caplog.at_level("WARNING"):
        state._check_forcing_null_projection(np.array([1.0, 0.0]))
        state._check_forcing_null_projection(np.array([1.0, 0.0]))

    matches = [
        record.message
        for record in caplog.records
        if "Coupled forcing projects strongly onto near-null modes" in record.message
    ]
    assert len(matches) == 1


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_expm_multiply_not_supported_for_jax_backend(tmp_path):
    from pynamit.simulation.runner import run_pynamit

    with pytest.raises(NotImplementedError, match="expm_multiply"):
        run_pynamit(
            run_directory=str(tmp_path / "exp_jax_expm_multiply"),
            final_time=1.0,
            dt=1.0,
            plotsteps=1,
            Nmax=5,
            Mmax=2,
            Ncs=6,
            dynamics_mode=DynamicsMode.FULL_INDUCTION,
            simulation_mode=SimulationMode.PURE_SPECTRAL,
            least_squares_solver="svd",
            integrator=IntegratorKind.EXPONENTIAL,
            dense_full_operators=False,
            exponential_solver=ExponentialSolverKind.EXPM_MULTIPLY,
        )
