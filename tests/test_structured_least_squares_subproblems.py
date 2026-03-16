from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pynamit.math.structured_least_squares import (
    ConstrainedStructuredLeastSquaresSubproblem,
    StructuredLeastSquaresSubproblem,
)
from pynamit.primitives.field import Field
from pynamit.simulation.dynamics import Dynamics
from pynamit.simulation.induction.toroidal_solver import DtAlphaSolveSystem
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, SimulationMode


def _seed_constant_conductance(dynamics: Dynamics) -> None:
    grid = dynamics.state.geometry.grid
    n_points = int(np.asarray(grid.lat).size)
    zeros = np.zeros(n_points, dtype=float)
    dynamics.state.etaP = Field.from_grid_values(
        grid, np.full(n_points, 1.0, dtype=float), zeros, zeros
    )
    dynamics.state.etaH = Field.from_grid_values(
        grid, np.full(n_points, 0.5, dtype=float), zeros, zeros
    )
    dynamics.state._invalidate_caches()


def _build_dynamics(
    tmp_path: Path,
    *,
    dynamics_mode: DynamicsMode,
    simulation_mode: SimulationMode,
    connect_hemispheres: bool = False,
    ll_constraint_mode: str = "auto",
) -> Dynamics:
    dynamics = Dynamics(
        run_directory=str(tmp_path / (f"{dynamics_mode}_{simulation_mode}_{ll_constraint_mode}")),
        Nmax=4,
        Mmax=1,
        Ncs=10,
        dynamics_mode=dynamics_mode,
        simulation_mode=simulation_mode,
        connect_hemispheres=connect_hemispheres,
        ll_constraint_mode=ll_constraint_mode,
        least_squares_solver="svd",
        integrator=IntegratorKind.EULER,
        benchmark_mode=True,
    )
    if dynamics_mode == DynamicsMode.LEGACY:
        _seed_constant_conductance(dynamics)
    return dynamics


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_legacy_and_full_induction_share_structured_subproblem_framework(tmp_path: Path) -> None:
    sim_legacy = _build_dynamics(
        tmp_path, dynamics_mode=DynamicsMode.LEGACY, simulation_mode=SimulationMode.CS_DOMINANT
    )
    assert isinstance(
        sim_legacy.state.m_imp_feedback_system.subproblem, StructuredLeastSquaresSubproblem
    )
    assert isinstance(
        sim_legacy.state.m_imp_feedback_system.solve_system,
        ConstrainedStructuredLeastSquaresSubproblem,
    )

    sim_full = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
    )
    solve_bundle = sim_full.state.toroidal_matrices.solver._get_dtalpha_solve_system(
        weighting=sim_full.state.toroidal_weighting,
        regularization_lambda=sim_full.state.toroidal_regularization_lambda,
        penalty_operator=None,
        penalty_scaling=0.0,
    )
    assert isinstance(solve_bundle, DtAlphaSolveSystem)
    assert isinstance(solve_bundle.solve_system, ConstrainedStructuredLeastSquaresSubproblem)
    assert isinstance(solve_bundle.subproblem, StructuredLeastSquaresSubproblem)
    assert solve_bundle.problem is solve_bundle.solve_system.problem
    assert solve_bundle.subproblem is solve_bundle.solve_system.subproblem


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_legacy_ll_constraint_modes_wire_soft_vs_hard(tmp_path: Path) -> None:
    sim_soft = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.LEGACY,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="soft",
    )
    sim_hard = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.LEGACY,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="hard",
    )

    assert sim_soft.state.m_imp_feedback_system.solve_system.equality_operator is None
    assert sim_soft.state.m_imp_feedback_system.problem.num_data_terms >= 2

    assert sim_hard.state.m_imp_feedback_system.solve_system.equality_operator is not None
    assert sim_hard.state.m_imp_feedback_system.problem.num_data_terms == 1


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_legacy_ll_constraint_mode_off_disables_hemisphere_constraint(tmp_path: Path) -> None:
    sim_off = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.LEGACY,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="off",
    )

    assert sim_off.state.m_imp_feedback_system.solve_system.equality_operator is None
    assert sim_off.state.m_imp_feedback_system.problem.num_data_terms == 1


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_ll_constraint_modes_split_hard_vs_soft_operators(tmp_path: Path) -> None:
    sim_hard = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="hard",
    )
    sim_soft = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="soft",
    )
    hard_system = sim_hard.state.dt_alpha_constraint_system
    soft_system = sim_soft.state.dt_alpha_constraint_system

    assert hard_system.hard_operator is not None
    assert hard_system.soft_operator is None

    assert soft_system.hard_operator is None
    assert soft_system.soft_operator is not None
    assert soft_system.soft_scaling > 0.0


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_ll_constraint_mode_off_disables_ll_rows(tmp_path: Path) -> None:
    sim_off = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="off",
    )
    off_system = sim_off.state.dt_alpha_constraint_system

    assert off_system.soft_operator is None
    assert off_system.hard_operator is None
    assert off_system.c_ll.shape[0] > 0


@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_full_induction_soft_ll_matches_direct_dtalpha_solve(tmp_path: Path) -> None:
    sim = _build_dynamics(
        tmp_path,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        connect_hemispheres=True,
        ll_constraint_mode="soft",
    )
    state = sim.state
    solver = state.toroidal_matrices.solver
    constraint_system = state.dt_alpha_constraint_system

    penalty_operator = constraint_system.soft_operator
    assert penalty_operator is not None
    penalty_scaling = float(constraint_system.soft_scaling)
    assert penalty_scaling > 0.0

    n_coeff = state.solution_space.index_length
    rhs_physics = np.linspace(0.25, 1.25, n_coeff, dtype=float)
    penalty_rhs = np.linspace(0.1, 1.0, penalty_operator.shape[0], dtype=float)
    constraint_operator = constraint_system.hard_operator
    rhs_constraint = (
        np.zeros(constraint_operator.shape[0], dtype=float)
        if constraint_operator is not None
        else np.zeros(0, dtype=float)
    )

    solved = np.asarray(
        solver.solve_dt_psi_superposed(
            rhs_physics=rhs_physics,
            rhs_constraint=rhs_constraint,
            constraint_operator=constraint_operator,
            m_imp_to_jr_operator=state.poloidal_matrices.m_imp_to_jr,
            weighting=state.toroidal_weighting,
            regularization_lambda=state.toroidal_regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            penalty_rhs=penalty_rhs,
            hinv_rtol=0.0,
            apply_psi_gauge=state.apply_psi_gauge,
        )
    ).reshape(-1)

    dtalpha = np.asarray(
        solver._solve_dtalpha_problem(
            rhs_physics_coeffs=rhs_physics,
            weighting=state.toroidal_weighting,
            regularization_lambda=state.toroidal_regularization_lambda,
            penalty_operator=penalty_operator,
            penalty_scaling=penalty_scaling,
            penalty_rhs=penalty_rhs,
            hinv_rtol=0.0,
            equality_operator=constraint_operator,
            equality_rhs=rhs_constraint if constraint_operator is not None else None,
        )
    ).reshape(-1)
    dtalpha_to_dt_psi = np.asarray(
        solver._get_dtalpha_to_dt_psi_map_cached(
            m_imp_to_jr_operator=state.poloidal_matrices.m_imp_to_jr,
            apply_psi_gauge=state.apply_psi_gauge,
        )
    )
    expected = np.asarray(dtalpha_to_dt_psi @ dtalpha).reshape(-1)

    np.testing.assert_allclose(solved, expected, atol=1e-12, rtol=1e-12)
