"""Targeted toroidal operator checks for the LaTeX-consistent formulation."""

from __future__ import annotations

import numpy as np
import pytest

import pynamit.simulation.induction.radial_shell_response as radial_shell_response_module
from pynamit.math.constants import RE, mu0
from pynamit.primitives.basis import get_repo_cf_helmholtz_sign, get_repo_df_helmholtz_sign
from pynamit.primitives.field import Field
from pynamit.simulation.spatial import to_dense
from pynamit.utils import to_numpy
from pynamit.simulation.induction import (
    AdditiveKnownSourceTraceModel,
    ColumnSolveKnownSourceTraceModel,
    CurrentFirstParticularPlusHarmonicKnownSourceTraceModel,
    EquivalentNonlocalExteriorToroidalUpdateModel,
    EquivalentNonlocalRadialShellResponseModel,
    CurrentContinuityExteriorToroidalUpdateModel,
    ExteriorToroidalScalarRadialResponseModel,
    FilteredKnownSourceTraceModel,
    FrozenConductanceIncrementalKnownElectricRadialResponseModel,
    GapCoenergyBlocks,
    HarmonicPoloidalSideTraceModel,
    HarmonicShellElectricTraceModel,
    HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel,
    InductivePlusHarmonicKnownSourceTraceModel,
    IncrementalIdealityCorrectedExteriorToroidalUpdateModel,
    KnownSourceOperatorKnownSourceTraceModel,
    KnownSourceTraceModel,
    MagneticRMBoundaryOperators,
    NonlocalShellElectricRadialResponseModel,
    PFACNonlocalRadialShellResponseModel,
    PoloidalSideTraceKnownSourceTraceModel,
    ProjectedTangentialSecondTraceModel,
    QTraceKnownSourceRadialResponseModel,
    RadialShellCondensedOperators,
    ExteriorToroidalUpdateKnownSourceTraceModel,
    RMToroidalBoundaryUpdateModel,
    RadialShellResponseModel,
    SchurComplementGapKnownSourceTraceModel,
    ShellCurrentContinuityKnownElectricRadialResponseModel,
    ShellCurrentDrivenKnownElectricRadialResponseModel,
    ShellElectricDifferenceKnownSourceTraceModel,
    ShellElectricTraceModel,
    ThinSheetCurrentContinuityKnownShellCurrentSourceModel,
    build_gap_coenergy_condensed_operators_from_blocks,
    build_known_source_operator_from_q_trace,
    build_q_trace_operator_from_known_source_model,
    build_q_trace_operator_from_exterior_update_model,
    build_q_trace_operator_from_poloidal_side_trace,
    build_radial_shell_rhs_from_q_trace_operator,
    build_radial_shell_rhs_from_trace_operator,
    build_projected_tangential_omitted_rhs_operator,
)
from pynamit.simulation.induction.operator_utils import coerce_dense_operator_matrix
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, MainfieldKind, SimulationMode


def _build_state(
    *,
    simulation_mode: SimulationMode = SimulationMode.PURE_SPECTRAL,
    nmax: int = 10,
    mmax: int = 5,
    ncs: int = 12,
    rm: float | None = None,
    magnetospheric_shielding: bool = True,
    toroidal_closure_mode: str = "radial_shell",
    radial_shell_response_model=None,
):
    if toroidal_closure_mode != "radial_shell":
        raise ValueError("Test helper supports only the canonical full_induction runtime.")
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=simulation_mode.value,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        RM=rm,
        magnetospheric_shielding=magnetospheric_shielding,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator=IntegratorKind.EULER,
        least_squares_solver="svd",
        radial_shell_response_model=radial_shell_response_model,
    )
    return sim.state


def _build_shadow_tangential_toroidal_matrices(state):
    model = EquivalentNonlocalRadialShellResponseModel()
    model.bind_state(state)
    return model._get_shadow_toroidal_matrices(state.toroidal_matrices)


class _SyntheticProjectedTangentialSecondTraceModel(ProjectedTangentialSecondTraceModel):
    def __init__(self, dense_op: np.ndarray):
        self.dense_op = np.asarray(dense_op, dtype=float)

    def build_second_trace_operator(self, toroidal_matrices):
        return self.dense_op


class _SyntheticShellElectricTraceModel(ShellElectricTraceModel):
    def __init__(self, dense_op: np.ndarray):
        self.dense_op = np.asarray(dense_op, dtype=float)

    def build_trace_operator(self, toroidal_matrices):
        return self.dense_op


class _SyntheticKnownSourceTraceModel(KnownSourceTraceModel):
    def __init__(self, dense_op: np.ndarray, dense_op_from_js: np.ndarray | None = None):
        self.dense_op = np.asarray(dense_op, dtype=float)
        self.dense_op_from_js = (
            None if dense_op_from_js is None else np.asarray(dense_op_from_js, dtype=float)
        )

    def build_q_trace_operator(self, toroidal_matrices):
        return self.dense_op

    def build_q_trace_from_js_operator(self, toroidal_matrices):
        return self.dense_op_from_js


class _SyntheticColumnSolveKnownSourceTraceModel(ColumnSolveKnownSourceTraceModel):
    pass


class _SyntheticKnownSourceResponseModel(RadialShellResponseModel):
    def __init__(self, gamma_known: np.ndarray):
        self.gamma_known = np.asarray(gamma_known, dtype=float)

    def build_known_source_operator(self, toroidal_matrices):
        return self.gamma_known


class _IdentityPoloidalSideTraceModel:
    def build_dudr_from_u_operator(self, toroidal_matrices):
        n = int(toroidal_matrices.basis.index_length)
        return np.eye(n, dtype=float)


def test_toroidal_forcing_gradient_field_is_zero() -> None:
    """Tangential-full forcing must vanish for pure cf (gradient) tangential E."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(7)
    phi = rng.normal(size=n)
    E_coeffs = np.vstack([phi, np.zeros_like(phi)])

    forcing = np.asarray(tor.compute_toroidal_rhs_from_E(E_coeffs), dtype=float).reshape(-1)
    assert np.linalg.norm(forcing) < 1e-10


def test_toroidal_forcing_depends_only_on_df_content() -> None:
    """Tangential-full forcing should depend only on the df/toroidal shell-electric content."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(17)
    cf = rng.normal(size=n)
    df = rng.normal(size=n)
    full = np.vstack([cf, df])
    df_only = np.vstack([np.zeros_like(df), df])

    actual_full = np.asarray(tor.compute_toroidal_rhs_from_E(full), dtype=float).reshape(-1)
    actual_df = np.asarray(tor.compute_toroidal_rhs_from_E(df_only), dtype=float).reshape(-1)

    np.testing.assert_allclose(actual_full, actual_df, rtol=1e-10, atol=1e-10)


def test_toroidal_forcing_matches_manual_surface_curl_formula() -> None:
    """The projected block inside tangential-full forcing should match the surface-curl formula."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(19)
    E_coeffs = rng.normal(size=(2, n))

    actual = np.asarray(tor.compute_toroidal_rhs_from_E(E_coeffs), dtype=float).reshape(-1)
    actual_projected = actual[:n]

    P = np.asarray(to_dense(tor.projection_matrix), dtype=float)
    G_th = np.asarray(
        to_dense(state.basis.get_evaluation_matrix(tor.grid, derivative="theta")), dtype=float
    )
    G_ph = np.asarray(
        to_dense(state.basis.get_evaluation_matrix(tor.grid, derivative="phi")), dtype=float
    )
    Eth_grid, Eph_grid = state.basis.evaluate(E_coeffs, tor.grid, vector_type="tangential")
    Eth_grid = np.asarray(to_numpy(Eth_grid), dtype=float).reshape(-1)
    Eph_grid = np.asarray(to_numpy(Eph_grid), dtype=float).reshape(-1)
    Eth_coeffs = P @ Eth_grid
    Eph_coeffs = P @ Eph_grid

    dEth_ph = G_ph @ Eth_coeffs
    dEph_th = G_th @ Eph_coeffs
    theta_rad = np.deg2rad(np.asarray(to_numpy(tor.grid.theta), dtype=float).reshape(-1))
    sin_th = np.sin(theta_rad)
    sin_th_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
    cot_th = np.cos(theta_rad) / sin_th_safe
    curl_omega = dEph_th + cot_th * Eph_grid - dEth_ph

    curl_coeffs = P @ curl_omega
    dcurl_th = G_th @ curl_coeffs
    dcurl_ph = G_ph @ curl_coeffs
    B0th = np.asarray(to_numpy(tor.b_field.vec.theta), dtype=float).reshape(-1)
    B0ph = np.asarray(to_numpy(tor.b_field.vec.phi), dtype=float).reshape(-1)
    expected_grid = (1.0 / float(tor.RI) ** 2) * ((B0th * dcurl_ph) - (B0ph * dcurl_th))
    expected = P @ expected_grid

    np.testing.assert_allclose(actual_projected, expected, rtol=1e-10, atol=1e-10)


def test_projected_tangential_omitted_rhs_matches_manual_second_trace_formula() -> None:
    """The omitted projected-driver builder should match its exact second-trace form."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(23)
    second_trace_op = rng.normal(size=(2 * n, 2 * n))
    model = _SyntheticProjectedTangentialSecondTraceModel(second_trace_op)

    actual = np.asarray(
        build_projected_tangential_omitted_rhs_operator(tor, model), dtype=float
    )

    q_r_op = second_trace_op[:n]
    p_r_op = second_trace_op[n:]
    P = np.asarray(to_dense(tor.projection_matrix), dtype=float)
    G_th = np.asarray(
        to_dense(state.basis.get_evaluation_matrix(tor.grid, derivative="theta")), dtype=float
    )
    G_ph = np.asarray(
        to_dense(state.basis.get_evaluation_matrix(tor.grid, derivative="phi")), dtype=float
    )
    B0th = np.asarray(to_numpy(tor.b_field.vec.theta), dtype=float).reshape(-1)
    B0ph = np.asarray(to_numpy(tor.b_field.vec.phi), dtype=float).reshape(-1)
    expected_grid = (1.0 / float(tor.RI)) * (
        (B0th[:, None] * ((G_th @ q_r_op) + (G_ph @ p_r_op)))
        + (B0ph[:, None] * ((G_ph @ q_r_op) - (G_th @ p_r_op)))
    )
    expected = P @ expected_grid

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_radial_shell_rhs_from_trace_matches_manual_first_trace_formula() -> None:
    """The exact radial-shell trace builder should match ``Delta_S(E_r - d_r U)``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(21)
    trace_op = rng.normal(size=(2 * n, 2 * n))
    model = _SyntheticShellElectricTraceModel(trace_op)

    actual = np.asarray(build_radial_shell_rhs_from_trace_operator(tor, model), dtype=float)

    dudr_op = trace_op[:n]
    er_op = trace_op[n:]
    lap = np.asarray(to_dense(state.basis.get_laplacian_operator(r=state.RI)), dtype=float)
    expected = lap @ (er_op - dudr_op)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_known_source_operator_from_q_trace_matches_manual_formula() -> None:
    """The exact q-trace builder should match ``-(1/mu0) Delta_S q``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(25)
    q_trace_op = rng.normal(size=(n, 2 * n))
    model = _SyntheticKnownSourceTraceModel(q_trace_op)

    actual = np.asarray(build_known_source_operator_from_q_trace(tor, model), dtype=float)
    lap = np.asarray(to_dense(state.basis.get_laplacian_operator(r=state.RI)), dtype=float)
    expected = (-(1.0 / float(mu0))) * (lap @ q_trace_op)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(build_radial_shell_rhs_from_q_trace_operator(tor, model), dtype=float),
        float(mu0) * expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_column_solve_known_source_trace_model_assembles_exact_dq_matrix() -> None:
    """Column-solve q-trace assembly should recover the underlying dense matrix exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    rng = np.random.default_rng(29)
    d_q = rng.normal(size=(n, 2 * n))

    model = _SyntheticColumnSolveKnownSourceTraceModel(
        q_column_solver=lambda toroidal_matrices, e: d_q @ np.asarray(e, dtype=float).reshape(-1)
    )

    actual = np.asarray(model.build_q_trace_operator(tor), dtype=float)
    np.testing.assert_allclose(actual, d_q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(build_known_source_operator_from_q_trace(tor, model), dtype=float),
        np.asarray(build_known_source_operator_from_q_trace(tor, _SyntheticKnownSourceTraceModel(d_q)), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_column_solve_known_source_trace_model_can_assemble_js_trace_matrix() -> None:
    """Column-solve q-trace assembly should also support a direct current-first map."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    rng = np.random.default_rng(30)
    d_q_from_js = rng.normal(size=(n, 2 * n))

    model = _SyntheticColumnSolveKnownSourceTraceModel(
        q_column_solver=lambda toroidal_matrices, e: np.zeros(n, dtype=float),
        q_from_js_column_solver=lambda toroidal_matrices, js: d_q_from_js @ np.asarray(js, dtype=float).reshape(-1),
    )

    actual = np.asarray(model.build_q_trace_from_js_operator(tor), dtype=float)
    np.testing.assert_allclose(actual, d_q_from_js, rtol=1e-12, atol=1e-12)


def test_gap_coenergy_block_condensation_matches_manual_schur_formula() -> None:
    """Abstract gap co-energy blocks should condense to the Schur formulas exactly."""
    rng = np.random.default_rng(19)
    n_chi = 5
    n_gap = 4
    n_forcing = 2 * n_chi

    K_cc = rng.normal(size=(n_chi, n_chi))
    K_cc = K_cc.T @ K_cc
    K_cx = rng.normal(size=(n_chi, n_gap))
    K_xx = rng.normal(size=(n_gap, n_gap))
    K_xx = K_xx.T @ K_xx + 0.5 * np.eye(n_gap)
    M_c = rng.normal(size=(n_chi, n_forcing))
    M_x = rng.normal(size=(n_gap, n_forcing))
    blocks = GapCoenergyBlocks(K_chichi=K_cc, K_chix=K_cx, K_xx=K_xx, M_chi=M_c, M_x=M_x)

    lambda_gap, gamma_known = build_gap_coenergy_condensed_operators_from_blocks(blocks)
    inv_kxx = np.linalg.pinv(K_xx)

    np.testing.assert_allclose(
        lambda_gap,
        K_cc - (K_cx @ inv_kxx @ K_cx.T),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        gamma_known,
        M_c - (K_cx @ inv_kxx @ M_x),
        rtol=1e-12,
        atol=1e-12,
    )


def test_gap_coenergy_q_trace_model_recovers_condensed_source_operator() -> None:
    """Schur-complement q-trace model should recover ``Gamma_known`` exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(tor.basis.index_length)
    rng = np.random.default_rng(23)
    n_gap = max(3, n // 4)

    K_cc = rng.normal(size=(n, n))
    K_cc = K_cc.T @ K_cc
    K_cx = rng.normal(size=(n, n_gap))
    K_xx = rng.normal(size=(n_gap, n_gap))
    K_xx = K_xx.T @ K_xx + np.eye(n_gap)
    M_c = rng.normal(size=(n, 2 * n))
    M_x = rng.normal(size=(n_gap, 2 * n))
    blocks = GapCoenergyBlocks(K_chichi=K_cc, K_chix=K_cx, K_xx=K_xx, M_chi=M_c, M_x=M_x)

    _, gamma_known = build_gap_coenergy_condensed_operators_from_blocks(blocks)
    q_model = SchurComplementGapKnownSourceTraceModel(blocks=blocks)

    np.testing.assert_allclose(
        np.asarray(build_known_source_operator_from_q_trace(tor, q_model), dtype=float),
        gamma_known,
        rtol=1e-10,
        atol=1e-10,
    )


def test_default_gap_coenergy_blocks_realize_condensed_known_source_model() -> None:
    """A known-source model should admit an exact zero-internal gap realization."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(tor.basis.index_length)
    rng = np.random.default_rng(31)
    gamma_known = rng.normal(size=(n, 2 * n))

    model = _SyntheticKnownSourceResponseModel(gamma_known)
    blocks = model.build_gap_coenergy_blocks(tor)
    assert blocks is not None
    assert blocks.K_chix.shape == (n, 0)
    assert blocks.K_xx.shape == (0, 0)
    assert blocks.M_x.shape == (0, 2 * n)

    lambda_gap, gamma_from_blocks = build_gap_coenergy_condensed_operators_from_blocks(blocks)

    np.testing.assert_allclose(gamma_from_blocks, gamma_known, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        lambda_gap,
        np.asarray(model.build_lambda_gap_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )

    q_model = SchurComplementGapKnownSourceTraceModel(blocks=blocks)
    np.testing.assert_allclose(
        np.asarray(build_known_source_operator_from_q_trace(tor, q_model), dtype=float),
        gamma_known,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "simulation_mode,nmax,mmax,ncs",
    [
        (SimulationMode.PURE_SPECTRAL, 8, 4, 10),
        (SimulationMode.SPECTRAL_TRANSFORM_GL, 8, 4, 10),
        (SimulationMode.SPECTRAL_TRANSFORM_CS, 6, 3, 8),
    ],
)
def test_equivalent_nonlocal_response_uses_nondegenerate_shell_pi_gap_blocks(
    simulation_mode: SimulationMode,
    nmax: int,
    mmax: int,
    ncs: int,
) -> None:
    """The condensed-inductive branch should expose ``Pi_shell`` as a real gap state."""
    state = _build_state(
        simulation_mode=simulation_mode,
        nmax=nmax,
        mmax=mmax,
        ncs=ncs,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices
    n = int(tor.basis.index_length)

    model = EquivalentNonlocalRadialShellResponseModel()
    model.bind_state(state)
    blocks = model.build_gap_coenergy_blocks(tor)
    assert blocks is not None
    assert blocks.K_chix.shape == (n, n)
    assert blocks.K_xx.shape == (n, n)
    assert blocks.M_x.shape == (n, 2 * n)

    shell_pi = np.asarray(
        state.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float
    )
    np.testing.assert_allclose(blocks.K_xx, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(-np.asarray(blocks.K_chix, dtype=float).T, shell_pi, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(blocks.M_x, 0.0, rtol=1e-12, atol=1e-12)

    lambda_gap_from_blocks, gamma_known_from_blocks = build_gap_coenergy_condensed_operators_from_blocks(
        blocks
    )
    np.testing.assert_allclose(
        lambda_gap_from_blocks,
        np.asarray(model.build_lambda_gap_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        gamma_known_from_blocks,
        np.asarray(model.build_gamma_known_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_current_first_response_uses_nondegenerate_shell_pi_gap_blocks() -> None:
    """The current-first forcing branch should also expose ``Pi_shell`` as gap state."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        nmax=8,
        mmax=4,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
    )
    tor = state.toroidal_matrices
    n = int(tor.basis.index_length)

    model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    model.bind_state(state)
    blocks = model.build_gap_coenergy_blocks(tor)
    assert blocks is not None
    assert blocks.K_chix.shape == (n, n)
    assert blocks.K_xx.shape == (n, n)
    assert blocks.M_x.shape == (n, 2 * n)

    shell_pi = np.asarray(
        state.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float
    )
    np.testing.assert_allclose(blocks.K_xx, np.eye(n), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        -np.asarray(blocks.K_chix, dtype=float).T, shell_pi, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(blocks.M_x, 0.0, rtol=1e-12, atol=1e-12)

    lambda_gap_from_blocks, gamma_known_from_blocks = build_gap_coenergy_condensed_operators_from_blocks(
        blocks
    )
    np.testing.assert_allclose(
        lambda_gap_from_blocks,
        np.asarray(model.build_lambda_gap_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        gamma_known_from_blocks,
        np.asarray(model.build_gamma_known_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_known_source_q_trace_adapter_matches_difference_of_first_traces() -> None:
    """The q-trace adapter should expose ``d_r U - E_r`` from first traces exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    first_trace_model = HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    q_model = ShellElectricDifferenceKnownSourceTraceModel(first_trace_model)

    first_trace_op = np.asarray(first_trace_model.build_trace_operator(tor), dtype=float)
    q_op = np.asarray(q_model.build_q_trace_operator(tor), dtype=float)

    np.testing.assert_allclose(q_op, first_trace_op[:n] - first_trace_op[n:], rtol=1e-12, atol=1e-12)


def test_filtered_known_source_trace_model_projects_requested_shell_channel() -> None:
    """Filtered q-trace models should zero the unrequested shell Helmholtz channel."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    rng = np.random.default_rng(7)
    base_op = rng.normal(size=(n, 2 * n))

    cf_model = FilteredKnownSourceTraceModel(
        base_q_trace_model=_SyntheticKnownSourceTraceModel(base_op),
        shell_channel="cf",
    )
    df_model = FilteredKnownSourceTraceModel(
        base_q_trace_model=_SyntheticKnownSourceTraceModel(base_op),
        shell_channel="df",
    )

    cf_op = np.asarray(cf_model.build_q_trace_operator(tor), dtype=float)
    df_op = np.asarray(df_model.build_q_trace_operator(tor), dtype=float)

    np.testing.assert_allclose(cf_op[:, n:], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df_op[:, :n], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cf_op + df_op, base_op, rtol=1e-12, atol=1e-12)


def test_filtered_known_source_trace_model_projects_requested_js_channel() -> None:
    """Filtered q-trace models should also project the requested channel in ``J_S`` space."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    rng = np.random.default_rng(17)
    base_from_js = rng.normal(size=(n, 2 * n))

    cf_model = FilteredKnownSourceTraceModel(
        base_q_trace_model=_SyntheticKnownSourceTraceModel(
            np.zeros((n, 2 * n), dtype=float), dense_op_from_js=base_from_js
        ),
        shell_channel="cf",
        input_space="js",
    )
    df_model = FilteredKnownSourceTraceModel(
        base_q_trace_model=_SyntheticKnownSourceTraceModel(
            np.zeros((n, 2 * n), dtype=float), dense_op_from_js=base_from_js
        ),
        shell_channel="df",
        input_space="js",
    )

    cf_op = np.asarray(cf_model.build_q_trace_from_js_operator(tor), dtype=float)
    df_op = np.asarray(df_model.build_q_trace_from_js_operator(tor), dtype=float)

    np.testing.assert_allclose(cf_op[:, n:], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(df_op[:, :n], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cf_op + df_op, base_from_js, rtol=1e-12, atol=1e-12)


def test_additive_known_source_trace_model_sums_q_trace_components() -> None:
    """Additive q-trace wrapper should sum component operators exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    rng = np.random.default_rng(11)
    op_a = rng.normal(size=(n, 2 * n))
    op_b = rng.normal(size=(n, 2 * n))

    model = AdditiveKnownSourceTraceModel(
        q_trace_models=(
            _SyntheticKnownSourceTraceModel(op_a),
            _SyntheticKnownSourceTraceModel(op_b),
        )
    )

    np.testing.assert_allclose(
        np.asarray(model.build_q_trace_operator(tor), dtype=float),
        op_a + op_b,
        rtol=1e-12,
        atol=1e-12,
    )


def test_direct_poloidal_side_q_trace_matches_harmonic_first_trace_route() -> None:
    """The direct ``Lambda_U U - E_r`` closure should match the harmonic first-trace path."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    direct_q_model = PoloidalSideTraceKnownSourceTraceModel(
        HarmonicPoloidalSideTraceModel(outer_boundary_mode="open")
    )
    direct_q_model.bind_state(state)

    first_trace_q_model = ShellElectricDifferenceKnownSourceTraceModel(
        HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    )
    first_trace_q_model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(direct_q_model.build_q_trace_operator(tor), dtype=float),
        np.asarray(first_trace_q_model.build_q_trace_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_direct_poloidal_side_q_trace_supports_js_adaptation() -> None:
    """The direct side-trace closure should expose the same q-trace through ``J_S -> E``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    q_model = PoloidalSideTraceKnownSourceTraceModel(
        HarmonicPoloidalSideTraceModel(outer_boundary_mode="open")
    )
    q_model.bind_state(state)

    q_from_e = np.asarray(q_model.build_q_trace_operator(tor), dtype=float)
    q_from_js = np.asarray(q_model.build_q_trace_from_js_operator(tor), dtype=float)
    js_to_e = np.asarray(
        coerce_dense_operator_matrix(state.JS_to_E_coeffs, n_component_rows=2, n_cols=2 * n),
        dtype=float,
    )

    np.testing.assert_allclose(q_from_js, q_from_e @ js_to_e, rtol=1e-12, atol=1e-12)


def test_homogeneous_outer_magnetic_column_solve_matches_direct_shielded_q_trace() -> None:
    """Homogeneous-``R_M`` column solve should match the direct shielded harmonic baseline."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5, rm=10.0 * RE
    )
    tor = state.toroidal_matrices

    direct_q_model = PoloidalSideTraceKnownSourceTraceModel(
        HarmonicPoloidalSideTraceModel(outer_boundary_mode="shielded")
    )
    direct_q_model.bind_state(state)

    column_model = HomogeneousOuterMagneticBoundaryColumnSolveKnownSourceTraceModel(
        outer_boundary_mode="shielded"
    )
    column_model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(column_model.build_q_trace_operator(tor), dtype=float),
        np.asarray(direct_q_model.build_q_trace_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(column_model.build_q_trace_from_js_operator(tor), dtype=float),
        np.asarray(direct_q_model.build_q_trace_from_js_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pragmatic_homogeneous_rm_connector_report_sums_current_runtime_components() -> None:
    """Pragmatic connector report should use the current upstream forcing split consistently."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5, rm=10.0 * RE
    )
    n = int(state.solution_space.index_length)
    rng = np.random.default_rng(1234)

    state.u = Field.from_coefficients(
        state.solution_space,
        coeffs=rng.normal(size=(2, n)),
        field_type="tangential",
    )
    state.Br = Field.from_coefficients(
        state.solution_space,
        coeffs=rng.normal(size=n),
        field_type="scalar",
    )
    state.jr = Field.from_coefficients(
        state.solution_space,
        coeffs=rng.normal(size=n),
        field_type="scalar",
    )
    state.m_imp_imposed = None
    state._imposed_toroidal_dirty = True

    report = state.get_pragmatic_homogeneous_rm_connector_report(
        outer_boundary_mode="shielded"
    )
    chi_report = state.get_pragmatic_homogeneous_rm_chi_report(
        outer_boundary_mode="shielded"
    )
    component_report = report["component_report"]

    np.testing.assert_allclose(
        np.asarray(chi_report["known_total_chi"], dtype=float),
        np.asarray(report["known_total_chi"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )

    q_total = np.asarray(component_report["total_external"]["q"], dtype=float)
    q_sum = (
        np.asarray(component_report["wind"]["q"], dtype=float)
        + np.asarray(component_report["Br"]["q"], dtype=float)
        + np.asarray(component_report["magnetic_imposed"]["q"], dtype=float)
    )
    np.testing.assert_allclose(q_total, q_sum, rtol=1e-12, atol=1e-12)

    chi_total = np.asarray(component_report["total_external"]["chi"], dtype=float)
    chi_sum = (
        np.asarray(component_report["wind"]["chi"], dtype=float)
        + np.asarray(component_report["Br"]["chi"], dtype=float)
        + np.asarray(component_report["magnetic_imposed"]["chi"], dtype=float)
    )
    np.testing.assert_allclose(chi_total, chi_sum, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q_total, float(state.RI) * chi_total, rtol=1e-12, atol=1e-12)

    dtjr_total = np.asarray(component_report["total_external"]["dtjr_known"], dtype=float)
    dtjr_sum = (
        np.asarray(component_report["wind"]["dtjr_known"], dtype=float)
        + np.asarray(component_report["Br"]["dtjr_known"], dtype=float)
        + np.asarray(component_report["magnetic_imposed"]["dtjr_known"], dtype=float)
    )
    np.testing.assert_allclose(dtjr_total, dtjr_sum, rtol=1e-12, atol=1e-12)


def test_pragmatic_homogeneous_rm_connector_report_exposes_driver_channel() -> None:
    """Pragmatic connector report should include the explicit ``dt_alpha`` driver piece."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5, rm=10.0 * RE
    )
    n = int(state.solution_space.index_length)
    rng = np.random.default_rng(5678)
    dt_m_imp = rng.normal(size=n)
    state.dt_m_imp_driver = Field.from_coefficients(
        state.solution_space,
        coeffs=dt_m_imp,
        field_type="scalar",
    )

    report = state.get_pragmatic_homogeneous_rm_connector_report(
        outer_boundary_mode="shielded"
    )
    driver_report = report["driver_report"]
    assert driver_report is not None
    assert report["chi_operator_norm"] == pytest.approx(report["q_operator_norm"] / float(state.RI))

    dt_alpha_driver = np.asarray(state._get_dt_alpha_driver_coeffs(), dtype=float).reshape(-1)
    alpha_to_psi = np.asarray(to_numpy(state.toroidal_matrices.alpha_to_psi_coeff_operator), dtype=float)
    expected_chi_driver = alpha_to_psi @ dt_alpha_driver

    np.testing.assert_allclose(
        np.asarray(driver_report["dt_alpha_driver"], dtype=float),
        dt_alpha_driver,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(driver_report["chi_driver"], dtype=float),
        expected_chi_driver,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(driver_report["q_driver"], dtype=float),
        float(state.RI) * expected_chi_driver,
        rtol=1e-12,
        atol=1e-12,
    )


def test_inductive_plus_harmonic_known_source_trace_matches_component_sum() -> None:
    """Experimental ``q_part + q_hom`` model should equal its filtered component sum."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    model = InductivePlusHarmonicKnownSourceTraceModel(outer_boundary_mode="open")
    model.bind_state(state)

    particular = FilteredKnownSourceTraceModel(
        base_q_trace_model=ExteriorToroidalUpdateKnownSourceTraceModel(
            EquivalentNonlocalExteriorToroidalUpdateModel()
        ),
        shell_channel="df",
    )
    particular.bind_state(state)
    homogeneous = FilteredKnownSourceTraceModel(
        base_q_trace_model=PoloidalSideTraceKnownSourceTraceModel(
            HarmonicPoloidalSideTraceModel(outer_boundary_mode="open")
        ),
        shell_channel="cf",
    )
    homogeneous.bind_state(state)

    expected = (
        np.asarray(particular.build_q_trace_operator(tor), dtype=float)
        + np.asarray(homogeneous.build_q_trace_operator(tor), dtype=float)
    )
    actual = np.asarray(model.build_q_trace_operator(tor), dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_current_first_particular_plus_harmonic_known_source_trace_matches_component_sum() -> None:
    """Current-first ``q_part + q_hom`` model should equal its filtered JS-space sum."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    model = CurrentFirstParticularPlusHarmonicKnownSourceTraceModel(outer_boundary_mode="open")
    model.bind_state(state)

    current_driven_response = ShellCurrentDrivenKnownElectricRadialResponseModel(
        shell_current_source_model=ThinSheetCurrentContinuityKnownShellCurrentSourceModel()
    )
    particular = FilteredKnownSourceTraceModel(
        base_q_trace_model=KnownSourceOperatorKnownSourceTraceModel(
            known_source_model=current_driven_response
        ),
        shell_channel="df",
        input_space="js",
    )
    particular.bind_state(state)
    homogeneous = FilteredKnownSourceTraceModel(
        base_q_trace_model=PoloidalSideTraceKnownSourceTraceModel(
            HarmonicPoloidalSideTraceModel(outer_boundary_mode="open")
        ),
        shell_channel="cf",
        input_space="js",
    )
    homogeneous.bind_state(state)

    expected = (
        np.asarray(particular.build_q_trace_from_js_operator(tor), dtype=float)
        + np.asarray(homogeneous.build_q_trace_from_js_operator(tor), dtype=float)
    )
    actual = np.asarray(model.build_q_trace_from_js_operator(tor), dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(
        np.asarray(particular.build_q_trace_from_js_operator(tor), dtype=float)
    ) < 1e-30


def test_q_trace_known_source_response_model_matches_exact_q_builder() -> None:
    """A q-trace forcing model should assemble the exact radial-shell source from q."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    q_model = ShellElectricDifferenceKnownSourceTraceModel(
        HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    )
    response_model = QTraceKnownSourceRadialResponseModel(q_trace_model=q_model)
    response_model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(response_model.build_known_source_operator(tor), dtype=float),
        np.asarray(build_known_source_operator_from_q_trace(tor, q_model), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_q_trace_from_equivalent_exterior_update_matches_manual_ri_dtpsi() -> None:
    """The non-harmonic q-trace adapter should expose ``q = R_I dtpsi^+`` exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    update_model = EquivalentNonlocalExteriorToroidalUpdateModel()
    update_model.bind_state(state)
    q_model = ExteriorToroidalUpdateKnownSourceTraceModel(update_model)
    q_model.bind_state(state)

    actual = np.asarray(q_model.build_q_trace_operator(tor), dtype=float)
    expected = float(state.RI) * np.asarray(update_model.build_dtpsi_operator(tor), dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual,
        np.asarray(build_q_trace_operator_from_exterior_update_model(tor, update_model), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_q_trace_from_equivalent_exterior_update_reproduces_equivalent_forcing_operator() -> None:
    """The q-trace route should reproduce the live equivalent nonlocal forcing scalar."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    equivalent_response_model = EquivalentNonlocalRadialShellResponseModel()
    equivalent_response_model.bind_state(state)
    q_model = ExteriorToroidalUpdateKnownSourceTraceModel(
        EquivalentNonlocalExteriorToroidalUpdateModel()
    )
    q_model.bind_state(state)
    q_response_model = QTraceKnownSourceRadialResponseModel(q_trace_model=q_model)
    q_response_model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(q_response_model.build_known_source_operator(tor), dtype=float),
        np.asarray(equivalent_response_model.build_known_source_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(q_response_model.build_rhs_operator(tor), dtype=float),
        np.asarray(equivalent_response_model.build_rhs_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(q_response_model.build_rhs_operator(tor), dtype=float),
        np.asarray(build_radial_shell_rhs_from_q_trace_operator(tor, q_model), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_q_trace_from_known_source_operator_matches_manual_ri_jr_to_psi() -> None:
    """The source-backed q adapter should expose ``q = R_I jr_to_psi dtj_r^+`` exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    source_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    source_model.bind_state(state)
    q_model = KnownSourceOperatorKnownSourceTraceModel(source_model)
    q_model.bind_state(state)

    actual = np.asarray(q_model.build_q_trace_operator(tor), dtype=float)
    expected = float(state.RI) * np.asarray(
        to_numpy(tor.jr_to_psi_coeff_operator),
        dtype=float,
    ) @ np.asarray(source_model.build_known_source_operator(tor), dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual,
        np.asarray(build_q_trace_operator_from_known_source_model(tor, source_model), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_current_first_shell_current_source_matches_manual_divergence_formula() -> None:
    """Current-first forcing law should be exactly ``-(1/R) div_Omega(dtK_S)``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    current_model = ThinSheetCurrentContinuityKnownShellCurrentSourceModel()
    current_model.bind_state(state)

    actual = np.asarray(current_model.build_known_source_from_js_operator(tor), dtype=float)
    expected = (-(1.0 / float(state.RI))) * np.asarray(
        coerce_dense_operator_matrix(
            state.solution_space.get_vector_divergence_operator(),
            n_cols=2 * int(state.solution_space.index_length),
        ),
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(current_model.build_q_trace_from_js_operator(tor), dtype=float),
        float(state.RI) * np.asarray(to_numpy(tor.jr_to_psi_coeff_operator), dtype=float) @ actual,
        rtol=1e-12,
        atol=1e-12,
    )


def test_frozen_conductance_source_factors_through_current_first_js_law() -> None:
    """Frozen-conductance forcing should equal current-first source composed with ``E -> J_S``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    model.bind_state(state)

    current_first = np.asarray(model.build_known_source_from_js_operator(tor), dtype=float)
    e_to_js = np.asarray(np.linalg.pinv(np.asarray(state.JS_to_E_coeffs.to_dense(), dtype=float)), dtype=float)
    expected = current_first @ e_to_js

    np.testing.assert_allclose(
        np.asarray(model.build_known_source_operator(tor), dtype=float),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(model.build_rhs_operator(tor), dtype=float),
        float(mu0) * expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_q_trace_from_known_source_operator_reproduces_incremental_forcing_operator() -> None:
    """The source-backed q route should reproduce the incremental forcing model exactly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    source_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    source_model.bind_state(state)
    q_model = KnownSourceOperatorKnownSourceTraceModel(source_model)
    q_model.bind_state(state)
    q_response_model = QTraceKnownSourceRadialResponseModel(q_trace_model=q_model)
    q_response_model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(q_response_model.build_known_source_operator(tor), dtype=float),
        np.asarray(source_model.build_known_source_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(q_response_model.build_rhs_operator(tor), dtype=float),
        np.asarray(source_model.build_rhs_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_radial_response_model_default_q_trace_inverts_known_source_operator() -> None:
    """Default radial-shell ``q`` builder should exactly invert the known source."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(model.build_q_trace_operator(tor), dtype=float),
        np.asarray(build_q_trace_operator_from_known_source_model(tor, model), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_exterior_scalar_radial_response_model_exposes_exact_q_trace() -> None:
    """Exterior-update radial-shell model should expose ``q`` through the one-curl update."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    update_model = EquivalentNonlocalExteriorToroidalUpdateModel()
    update_model.bind_state(state)
    model = ExteriorToroidalScalarRadialResponseModel(exterior_update_model=update_model)
    model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(model.build_q_trace_operator(tor), dtype=float),
        np.asarray(build_q_trace_operator_from_exterior_update_model(tor, update_model), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_radial_shell_mass_dtalpha_operator_matches_mu0_alpha_to_jr() -> None:
    """The canonical radial-shell mass block should be ``mu0 * alpha_to_jr``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices

    expected = mu0 * np.asarray(tor.alpha_to_jr_coeff_operator, dtype=float)
    actual = np.asarray(tor.radial_shell_mass_dtalpha_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_equivalent_nonlocal_radial_shell_response_matches_tangential_full_dtalpha_map() -> None:
    """Equivalent radial-shell response should reproduce the tangential-full ``E -> dt_alpha`` map."""
    equivalent_model = EquivalentNonlocalRadialShellResponseModel()
    radial_state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=equivalent_model,
    )

    tangential_tor = equivalent_model._get_shadow_toroidal_matrices(radial_state.toroidal_matrices)
    radial_tor = radial_state.toroidal_matrices
    radial_constraints = radial_state.dt_alpha_constraint_system

    tangential_dtalpha_from_e = np.asarray(
        tangential_tor.build_dtalpha_from_toroidal_rhs_matrix(
            constraint_operator=radial_constraints.hard_operator,
            weighting=radial_state.toroidal_weighting,
            regularization_lambda=radial_state.toroidal_regularization_lambda,
            penalty_operator=radial_constraints.soft_operator,
            penalty_scaling=float(radial_constraints.soft_scaling),
            hinv_rtol=0.0,
        ),
        dtype=float,
    ) @ np.asarray(tangential_tor.toroidal_rhs_from_E_operator, dtype=float)

    radial_dtalpha_from_e = np.asarray(
        radial_tor.build_dtalpha_from_toroidal_rhs_matrix(
            constraint_operator=radial_constraints.hard_operator,
            weighting=radial_state.toroidal_weighting,
            regularization_lambda=radial_state.toroidal_regularization_lambda,
            penalty_operator=radial_constraints.soft_operator,
            penalty_scaling=float(radial_constraints.soft_scaling),
            hinv_rtol=0.0,
        ),
        dtype=float,
    ) @ np.asarray(radial_tor.toroidal_rhs_from_E_operator, dtype=float)

    np.testing.assert_allclose(radial_dtalpha_from_e, tangential_dtalpha_from_e, rtol=1e-10, atol=1e-10)


def test_equivalent_nonlocal_radial_shell_response_defaults_to_tangential_full_shadow() -> None:
    """The canonical equivalent radial-shell model should use tangential_full as its shadow closure."""
    model = EquivalentNonlocalRadialShellResponseModel()

    assert model.benchmark_closure_mode == "tangential_full"


def test_equivalent_nonlocal_exterior_update_dtjr_and_dtpsi_forms_match() -> None:
    """`dtj_r^+` and `dtpsi^+` forms should induce the same radial-shell rhs."""
    update_model = EquivalentNonlocalExteriorToroidalUpdateModel()
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=ExteriorToroidalScalarRadialResponseModel(
            exterior_update_model=update_model
        ),
    )
    tor = state.toroidal_matrices

    dtjr_op = np.asarray(update_model.build_dtjr_operator(tor), dtype=float)
    dtpsi_op = np.asarray(update_model.build_dtpsi_operator(tor), dtype=float)
    lap = np.asarray(to_dense(state.basis.get_laplacian_operator(r=state.RI)), dtype=float)

    np.testing.assert_allclose(
        mu0 * dtjr_op,
        (-(float(state.RI))) * (lap @ dtpsi_op),
        rtol=1e-10,
        atol=1e-10,
    )


def test_exterior_toroidal_update_adapter_matches_equivalent_nonlocal_radial_rhs() -> None:
    """The explicit scalar-update adapter should reproduce the equivalent radial rhs."""
    equivalent_model = EquivalentNonlocalRadialShellResponseModel()
    scalar_model = ExteriorToroidalScalarRadialResponseModel(
        exterior_update_model=EquivalentNonlocalExteriorToroidalUpdateModel()
    )

    equivalent_state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=equivalent_model,
    )
    scalar_state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=scalar_model,
    )

    np.testing.assert_allclose(
        np.asarray(equivalent_state.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float),
        np.asarray(scalar_state.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_builtin_pfac_radial_shell_model_matches_explicit_manual_composition() -> None:
    """Built-in PFAC radial-shell model should match the explicit manual composition."""
    state_auto = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    state_manual = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=NonlocalShellElectricRadialResponseModel(
            shell_response_model=QTraceKnownSourceRadialResponseModel(
                q_trace_model=ExteriorToroidalUpdateKnownSourceTraceModel(
                    EquivalentNonlocalExteriorToroidalUpdateModel()
                )
            ),
            exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel(),
        ),
    )

    np.testing.assert_allclose(
        np.asarray(state_auto.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float),
        np.asarray(
            state_manual.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(
            state_auto.toroidal_matrices.full_radial_shell_feedback_dtalpha_operator, dtype=float
        ),
        np.asarray(
            state_manual.toroidal_matrices.full_radial_shell_feedback_dtalpha_operator, dtype=float
        ),
        rtol=1e-10,
        atol=1e-10,
    )


def test_builtin_pfac_radial_shell_model_uses_current_continuity_feedback() -> None:
    """Built-in PFAC radial-shell model should use the explicit continuity connector."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    trace_only = NonlocalShellElectricRadialResponseModel(
        shell_response_model=QTraceKnownSourceRadialResponseModel(
            q_trace_model=ExteriorToroidalUpdateKnownSourceTraceModel(
                EquivalentNonlocalExteriorToroidalUpdateModel()
            )
        )
    )
    trace_only.bind_state(state)

    feedback_only = ExteriorToroidalScalarRadialResponseModel(
        exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel()
    )
    feedback_only.bind_state(state)

    built_in = state.radial_shell_response_model
    np.testing.assert_allclose(
        np.asarray(built_in.build_rhs_operator(tor), dtype=float),
        np.asarray(trace_only.build_rhs_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_builtin_pfac_radial_shell_frozen_incremental_mode_matches_q_trace_composition() -> None:
    """Frozen-conductance forcing mode should also be routed through the exact q-trace path."""
    state_auto = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(
            forcing_mode="frozen_conductance_incremental"
        ),
    )
    state_manual = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=NonlocalShellElectricRadialResponseModel(
            shell_response_model=QTraceKnownSourceRadialResponseModel(
                q_trace_model=KnownSourceOperatorKnownSourceTraceModel(
                    FrozenConductanceIncrementalKnownElectricRadialResponseModel()
                )
            ),
            exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel(),
        ),
    )

    np.testing.assert_allclose(
        np.asarray(state_auto.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float),
        np.asarray(
            state_manual.toroidal_matrices.full_radial_shell_rhs_from_E_operator, dtype=float
        ),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(
            state_auto.toroidal_matrices.full_radial_shell_feedback_dtalpha_operator, dtype=float
        ),
        np.asarray(
            state_manual.toroidal_matrices.full_radial_shell_feedback_dtalpha_operator, dtype=float
        ),
        rtol=1e-10,
        atol=1e-10,
    )


def test_builtin_pfac_radial_shell_incremental_mode_exposes_q_trace_operator() -> None:
    """Incremental built-in forcing mode should expose the exact source-inverted q trace."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(
            forcing_mode="frozen_conductance_incremental"
        ),
    )
    tor = state.toroidal_matrices

    source_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    source_model.bind_state(state)
    expected = np.asarray(build_q_trace_operator_from_known_source_model(tor, source_model), dtype=float)
    actual = np.asarray(tor.full_radial_shell_q_trace_from_E_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        actual,
        np.asarray(state.radial_shell_response_model.build_q_trace_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_known_source_operator() -> None:
    """Built-in radial-shell forcing operator should expose ``dtj_r^+`` directly."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    actual = np.asarray(tor.full_radial_shell_known_source_from_E_operator, dtype=float)
    expected = (1.0 / float(mu0)) * np.asarray(tor.full_radial_shell_rhs_from_E_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    model = state.radial_shell_response_model
    np.testing.assert_allclose(
        actual,
        np.asarray(model.build_known_source_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_realized_gap_operators() -> None:
    """Built-in radial-shell forcing should expose the same realized gap operators."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices
    model = state.radial_shell_response_model

    blocks = model.build_gap_coenergy_blocks(tor)
    assert blocks is not None
    lambda_gap, gamma_known = build_gap_coenergy_condensed_operators_from_blocks(blocks)

    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_lambda_gap_operator, dtype=float),
        np.asarray(lambda_gap, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_known_source_from_E_operator, dtype=float),
        np.asarray(gamma_known, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_gamma_known_and_dq_aliases() -> None:
    """Theory-note aliases should match the existing forcing-side operators exactly."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_gamma_known_operator, dtype=float),
        np.asarray(tor.full_radial_shell_known_source_from_E_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_d_q_operator, dtype=float),
        np.asarray(tor.full_radial_shell_q_trace_from_E_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_unified_condensed_operator_bundle() -> None:
    """Built-in radial-shell forcing should expose the same operators through one bundle."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    bundle = tor.full_radial_shell_condensed_operators
    assert isinstance(bundle, RadialShellCondensedOperators)

    np.testing.assert_allclose(
        np.asarray(bundle.Lambda_gap, dtype=float),
        np.asarray(tor.full_radial_shell_lambda_gap_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle.Gamma_known, dtype=float),
        np.asarray(tor.full_radial_shell_known_source_from_E_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle.D_q, dtype=float),
        np.asarray(tor.full_radial_shell_q_trace_from_E_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle.Gamma_induced_dtalpha, dtype=float),
        np.asarray(tor.full_radial_shell_induced_source_dtalpha_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle.D_q_induced_dtalpha, dtype=float),
        np.asarray(tor.full_radial_shell_induced_q_trace_dtalpha_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_frozen_conductance_bundle_carries_current_first_operators() -> None:
    """Frozen-conductance bundle should preserve the optional current-first path."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(
            forcing_mode="frozen_conductance_incremental"
        ),
    )
    tor = state.toroidal_matrices
    bundle = tor.full_radial_shell_condensed_operators

    assert bundle.Gamma_known_from_js is not None
    assert bundle.D_q_from_js is not None
    np.testing.assert_allclose(
        np.asarray(bundle.Gamma_known_from_js, dtype=float),
        np.asarray(tor.full_radial_shell_known_source_from_JS_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle.D_q_from_js, dtype=float),
        np.asarray(tor.full_radial_shell_q_trace_from_JS_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_gamma_known_from_JS_operator, dtype=float),
        np.asarray(tor.full_radial_shell_known_source_from_JS_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_d_q_from_JS_operator, dtype=float),
        np.asarray(tor.full_radial_shell_q_trace_from_JS_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_condensed_bundle_carries_js_adapted_current_first_operators() -> None:
    """Default condensed forcing mode should expose the same canonical shell-current input."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices
    bundle = tor.full_radial_shell_condensed_operators

    assert bundle.Gamma_known_from_js is not None
    assert bundle.D_q_from_js is not None
    np.testing.assert_allclose(
        np.asarray(bundle.Gamma_known_from_js, dtype=float),
        np.asarray(tor.full_radial_shell_known_source_from_JS_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(bundle.D_q_from_js, dtype=float),
        np.asarray(tor.full_radial_shell_q_trace_from_JS_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_frozen_conductance_radial_shell_exposes_current_first_js_source_operator() -> None:
    """Frozen-conductance forcing mode should expose the primitive ``J_S -> dtj_r^+`` law."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(
            forcing_mode="frozen_conductance_incremental"
        ),
    )
    tor = state.toroidal_matrices

    explicit_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    explicit_model.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_known_source_from_JS_operator, dtype=float),
        np.asarray(explicit_model.build_known_source_from_js_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_q_trace_from_JS_operator, dtype=float),
        np.asarray(explicit_model.build_q_trace_from_js_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_q_trace_operator() -> None:
    """Built-in radial-shell forcing path should expose the exact shell q trace."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    actual = np.asarray(tor.full_radial_shell_q_trace_from_E_operator, dtype=float)
    update_model = EquivalentNonlocalExteriorToroidalUpdateModel()
    update_model.bind_state(state)
    expected = np.asarray(
        build_q_trace_operator_from_exterior_update_model(tor, update_model),
        dtype=float,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        actual,
        np.asarray(state.radial_shell_response_model.build_q_trace_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_js_adapted_forcing_operators() -> None:
    """Default condensed forcing mode should expose ``J_S -> dtj_r^+`` and ``J_S -> q``."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices
    n = int(tor.basis.index_length)

    js_to_e = np.asarray(
        coerce_dense_operator_matrix(state.JS_to_E_coeffs, n_component_rows=2, n_cols=2 * n),
        dtype=float,
    )

    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_known_source_from_JS_operator, dtype=float),
        np.asarray(tor.full_radial_shell_known_source_from_E_operator, dtype=float) @ js_to_e,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(tor.full_radial_shell_q_trace_from_JS_operator, dtype=float),
        np.asarray(tor.full_radial_shell_q_trace_from_E_operator, dtype=float) @ js_to_e,
        rtol=1e-12,
        atol=1e-12,
    )


def test_builtin_pfac_radial_shell_exposes_induced_source_operator() -> None:
    """Built-in radial-shell feedback should expose the induced ``dtj_r^+`` source."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    continuity_update = CurrentContinuityExteriorToroidalUpdateModel()
    continuity_update.bind_state(state)

    actual = np.asarray(tor.full_radial_shell_induced_source_dtalpha_operator, dtype=float)
    expected = np.asarray(continuity_update.build_dtjr_from_dtalpha_operator(tor), dtype=float)

    np.testing.assert_allclose(
        actual,
        (1.0 / float(mu0)) * np.asarray(tor.full_radial_shell_feedback_dtalpha_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual,
        np.asarray(state.radial_shell_response_model.build_induced_source_dtalpha_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_builtin_pfac_radial_shell_exposes_induced_q_trace_operator() -> None:
    """Built-in radial-shell feedback should expose the induced ``q`` trace."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    actual = np.asarray(tor.full_radial_shell_induced_q_trace_dtalpha_operator, dtype=float)
    expected = float(state.RI) * np.asarray(to_numpy(tor.jr_to_psi_coeff_operator), dtype=float) @ np.asarray(
        tor.full_radial_shell_induced_source_dtalpha_operator, dtype=float
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual,
        np.asarray(
            state.radial_shell_response_model.build_induced_q_trace_from_dtalpha_operator(tor),
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_toroidal_to_e_factorization_matches_direct_operator() -> None:
    """The explicit ``psi -> J_S -> E`` chain should match the direct ``psi -> E`` map."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)

    direct = np.asarray(state.toroidal_to_E_coeffs.to_dense(), dtype=float)
    factorized = np.asarray(state.JS_to_E_coeffs.to_dense(), dtype=float) @ np.asarray(
        state.toroidal_to_JS_coeffs.to_dense(), dtype=float
    )

    np.testing.assert_allclose(factorized, direct, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("shielding", [False, True])
def test_dynamic_toroidal_operator_chain_report_matches_full_composition(
    shielding: bool,
) -> None:
    """The explicit dynamic operator report should match the live coupled top-left block."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        magnetospheric_shielding=shielding,
    )
    report = state.get_dynamic_toroidal_operator_chain_report()
    pi_report = state.get_dynamic_toroidal_pi_report()

    n = int(state.solution_space.index_length)
    selected_pfac = np.asarray(state.poloidal_matrices.dynamic_toroidal_pfac_operator, dtype=float)
    selected_pi = np.asarray(state.poloidal_matrices.dynamic_toroidal_pi_operator, dtype=float)
    expected_selected = np.asarray(
        state.poloidal_matrices.dynamic_toroidal_pfac_closed_operator
        if shielding
        else state.poloidal_matrices.dynamic_toroidal_pfac_open_operator,
        dtype=float,
    )

    np.testing.assert_allclose(report["pi_like_dynamic_pfac_operator"], pi_report["pi_operator"])
    np.testing.assert_allclose(report["pi_like_dynamic_pfac_operator"], selected_pfac)
    np.testing.assert_allclose(pi_report["pi_operator"], selected_pi)
    np.testing.assert_allclose(pi_report["pi_effective_operator"], selected_pi)
    np.testing.assert_allclose(report["pi_like_dynamic_pfac_operator"], expected_selected)
    np.testing.assert_allclose(pi_report["pi_open_operator"], state.poloidal_matrices.dynamic_toroidal_pi_open_operator)
    np.testing.assert_allclose(
        pi_report["pi_closed_operator"], state.poloidal_matrices.dynamic_toroidal_pi_closed_operator
    )
    np.testing.assert_allclose(
        pi_report["pi_reaction_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_reaction_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_effective_operator"],
        pi_report["pi_open_operator"] + pi_report["pi_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_rm_boundary_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_rm_boundary_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_rm_boundary_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_rm_boundary_effective_operator"],
        pi_report["pi_rm_boundary_open_operator"] + pi_report["pi_rm_boundary_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_closed_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_closed_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_effective_operator"],
        pi_report["pi_to_br_open_operator"] + pi_report["pi_to_br_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_rm_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_rm_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_rm_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_rm_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_rm_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_br_rm_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_br_rm_effective_operator"],
        pi_report["pi_to_br_rm_open_operator"] + pi_report["pi_to_br_rm_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_closed_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_closed_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_effective_operator"],
        pi_report["pi_to_dbr_open_operator"] + pi_report["pi_to_dbr_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_rm_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_rm_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_rm_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_to_dbr_rm_effective_operator"],
        pi_report["pi_to_dbr_rm_open_operator"] + pi_report["pi_to_dbr_rm_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_harmonic_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_harmonic_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_harmonic_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_harmonic_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_to_br_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_to_br_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_to_br_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_to_br_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_to_br_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_to_br_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_to_dbr_open_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_open_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_to_dbr_effective_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_effective_operator,
    )
    np.testing.assert_allclose(
        pi_report["shell_pi_to_dbr_shielding_operator"],
        state.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_shielding_operator,
    )
    np.testing.assert_allclose(
        pi_report["pi_open_operator"],
        pi_report["shell_pi_harmonic_open_operator"] + pi_report["shell_pi_open_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_effective_operator"],
        pi_report["shell_pi_harmonic_effective_operator"] + pi_report["shell_pi_effective_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pi_report["pi_shielding_operator"],
        pi_report["shell_pi_harmonic_shielding_operator"] + pi_report["shell_pi_shielding_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        report["psi_to_js"], np.asarray(state.toroidal_to_JS_coeffs.to_dense(), dtype=float)
    )
    np.testing.assert_allclose(
        report["psi_to_e"], np.asarray(state.toroidal_to_E_coeffs.to_dense(), dtype=float)
    )
    np.testing.assert_allclose(
        report["psi_to_js_poloidal_block"],
        report["expected_psi_to_js_poloidal_block"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        report["psi_to_js_toroidal_block"],
        report["expected_psi_to_js_toroidal_block"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        report["psi_to_e_factorized"], report["psi_to_e"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        report["psi_to_dtpsi_from_rhs_chain"],
        report["psi_to_dtpsi"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        report["psi_to_dtpsi"], report["coupled_top_left"], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(pi_report["psi_to_js"], report["psi_to_js"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        pi_report["direct_poloidal_psi_to_js_operator"] + pi_report["pfac_pi_to_js_operator"],
        pi_report["psi_to_js"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pi_report["direct_poloidal_psi_to_e_operator"] + pi_report["pfac_pi_to_e_operator"],
        pi_report["psi_to_e"],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pi_report["direct_poloidal_psi_to_dtpsi_operator"]
        + pi_report["pfac_pi_to_dtpsi_operator"],
        pi_report["psi_to_dtpsi"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert pi_report["psi_to_e_pi_split_difference_norm"] < 1e-8
    assert pi_report["rhs_pi_split_difference_norm"] < 1e-10
    assert pi_report["psi_to_dtpsi_pi_split_difference_norm"] < 1e-10


def test_reduced_shell_closure_report_matches_runtime_shell_operators() -> None:
    """The reduced-shell ``q`` report should match the live ``chi`` operators exactly."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
    )
    report = state.get_reduced_shell_closure_report()
    tor = state.toroidal_matrices
    ri = float(tor.RI)

    np.testing.assert_allclose(
        report["q_to_shell_pi_operator"],
        (1.0 / ri)
        * np.asarray(state.poloidal_matrices.dynamic_toroidal_shell_pi_effective_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        report["q_to_shell_dbr_operator"],
        (1.0 / ri)
        * np.asarray(
            state.poloidal_matrices.dynamic_toroidal_shell_pi_to_dbr_effective_operator,
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        report["radial_source_from_q_operator"],
        (1.0 / ri)
        * np.asarray(state.radial_shell_response_model.build_lambda_gap_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        report["tangential_total_q_operator"],
        report["tangential_local_q_operator"] + report["tangential_return_q_operator"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert report["tangential_total_split_difference_norm"] <= 1e-12
    assert report["tangential_local_matches_fieldline_advection_norm"] <= 1e-9


def test_dynamic_toroidal_pi_radius_report_matches_boundary_specializations() -> None:
    """Arbitrary-radius ``Pi`` report should reduce to the explicit ``R_I``/``R_M`` operators."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        magnetospheric_shielding=True,
    )
    ri_report = state.get_dynamic_toroidal_pi_radius_report(float(state.RI))
    rm_report = state.get_dynamic_toroidal_pi_radius_report(float(state.RM))
    mid_radius = 0.5 * (float(state.RI) + float(state.RM))
    mid_report = state.get_dynamic_toroidal_pi_radius_report(mid_radius)

    np.testing.assert_allclose(
        ri_report["pi_open_total"], state.poloidal_matrices.dynamic_toroidal_pi_open_operator
    )
    np.testing.assert_allclose(
        ri_report["pi_effective_total"], state.poloidal_matrices.dynamic_toroidal_pi_effective_operator
    )
    np.testing.assert_allclose(
        ri_report["pi_to_dbr_effective_total"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_effective_operator,
    )

    np.testing.assert_allclose(
        rm_report["pi_open_total"], state.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_open_operator
    )
    np.testing.assert_allclose(
        rm_report["pi_effective_total"],
        state.poloidal_matrices.dynamic_toroidal_pi_rm_boundary_effective_operator,
    )
    np.testing.assert_allclose(
        rm_report["pi_to_dbr_effective_total"],
        state.poloidal_matrices.dynamic_toroidal_pi_to_dbr_rm_effective_operator,
    )

    assert mid_report["pi_open_split_difference_norm"] < 1e-12
    assert mid_report["pi_effective_split_difference_norm"] < 1e-12
    assert mid_report["pi_shielding_split_difference_norm"] < 1e-12
    assert mid_report["pi_to_br_open_split_difference_norm"] < 1e-12
    assert mid_report["pi_to_br_effective_split_difference_norm"] < 1e-12
    assert mid_report["pi_to_dbr_open_split_difference_norm"] < 1e-12
    assert mid_report["pi_to_dbr_effective_split_difference_norm"] < 1e-12


def test_shell_current_continuity_known_e_rhs_matches_manual_divergence_formula() -> None:
    """Explicit forcing-side current-continuity model should match its shell-law formula."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    model.bind_state(state)
    actual = np.asarray(model.build_rhs_operator(tor), dtype=float)

    js_to_e = np.asarray(state.JS_to_E_coeffs.to_dense(), dtype=float)
    rtol = max(float(getattr(state, "induction_null_svd_rtol", 0.0)), 0.0)
    if rtol <= 0.0:
        rtol = max(np.finfo(float).eps * max(js_to_e.shape), 1e-15)
    expected_e_to_js = np.asarray(np.linalg.pinv(js_to_e, rcond=rtol), dtype=float)
    div_omega = np.asarray(
        to_dense(state.solution_space.get_vector_divergence_operator()), dtype=float
    )
    expected = (-(float(mu0) / float(state.RI))) * (div_omega @ expected_e_to_js)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_shell_current_continuity_alias_matches_incremental_model() -> None:
    """Historical shell-current forcing name should remain an exact alias."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices

    incremental = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    incremental.bind_state(state)
    alias = ShellCurrentContinuityKnownElectricRadialResponseModel()
    alias.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(incremental.build_rhs_operator(tor), dtype=float),
        np.asarray(alias.build_rhs_operator(tor), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_shell_current_continuity_known_e_model_is_available_in_full_radial_shell() -> None:
    """Full-induction radial-shell should accept the explicit shell-continuity forcing model."""
    response_model = NonlocalShellElectricRadialResponseModel(
        shell_response_model=FrozenConductanceIncrementalKnownElectricRadialResponseModel(),
        exterior_update_model=CurrentContinuityExteriorToroidalUpdateModel(),
    )
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=response_model,
    )
    tor = state.toroidal_matrices
    rhs = np.asarray(tor.full_radial_shell_rhs_from_E_operator, dtype=float)
    feedback = np.asarray(tor.full_radial_shell_feedback_dtalpha_operator, dtype=float)

    n = int(state.solution_space.index_length)
    assert rhs.shape == (n, 2 * n)
    assert feedback.shape == (n, n)
    assert np.all(np.isfinite(rhs))
    assert np.all(np.isfinite(feedback))


def test_incremental_corrected_connector_matches_explicit_factorized_shell_chain() -> None:
    """The matched connector should equal the explicit ``psi -> J_S -> E`` shell chain."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    equivalent_shell_model = EquivalentNonlocalRadialShellResponseModel()
    equivalent_shell_model.bind_state(state)
    corrected_update = IncrementalIdealityCorrectedExteriorToroidalUpdateModel(
        shell_feedback_response_model=equivalent_shell_model,
        base_exterior_update_model=RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed"),
    )
    corrected_update.bind_state(state)

    rhs_from_e = np.asarray(equivalent_shell_model.build_rhs_operator(tor), dtype=float)
    js_to_e = np.asarray(state.JS_to_E_coeffs.to_dense(), dtype=float)
    psi_to_js = np.asarray(state.toroidal_to_JS_coeffs.to_dense(), dtype=float)
    alpha_to_psi = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)
    jr_to_psi = np.asarray(tor.jr_to_psi_coeff_operator, dtype=float)

    manual = jr_to_psi @ ((1.0 / float(mu0)) * (rhs_from_e @ js_to_e @ psi_to_js @ alpha_to_psi))
    matched = np.asarray(corrected_update.build_matched_dtpsi_from_dtalpha_operator(tor), dtype=float)

    np.testing.assert_allclose(matched, manual, rtol=1e-10, atol=1e-10)


def test_current_continuity_connector_matches_corrected_connector() -> None:
    """Thin-shell current continuity should reproduce the earlier corrected connector."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    tor = state.toroidal_matrices

    corrected_update = IncrementalIdealityCorrectedExteriorToroidalUpdateModel(
        shell_feedback_response_model=EquivalentNonlocalRadialShellResponseModel(),
        base_exterior_update_model=RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed"),
    )
    corrected_update.bind_state(state)
    continuity_update = CurrentContinuityExteriorToroidalUpdateModel()
    continuity_update.bind_state(state)

    np.testing.assert_allclose(
        np.asarray(continuity_update.build_dtjr_from_dtalpha_operator(tor), dtype=float),
        np.asarray(corrected_update.build_dtjr_from_dtalpha_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(continuity_update.build_dtpsi_from_dtalpha_operator(tor), dtype=float),
        np.asarray(corrected_update.build_dtpsi_from_dtalpha_operator(tor), dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


def test_radial_shell_feedback_comparison_report_matches_direct_operator_comparison() -> None:
    """Runtime report should reproduce the direct shell-vs-connector feedback comparison."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    report = state.get_radial_shell_feedback_comparison_report()
    tor = state.toroidal_matrices

    shell_feedback_model = NonlocalShellElectricRadialResponseModel(
        shell_response_model=EquivalentNonlocalRadialShellResponseModel()
    )
    shell_feedback_model.bind_state(state)
    raw_connector_feedback_model = ExteriorToroidalScalarRadialResponseModel(
        exterior_update_model=RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed")
    )
    raw_connector_feedback_model.bind_state(state)
    corrected_connector_feedback_model = ExteriorToroidalScalarRadialResponseModel(
        exterior_update_model=IncrementalIdealityCorrectedExteriorToroidalUpdateModel(
            shell_feedback_response_model=EquivalentNonlocalRadialShellResponseModel(),
            base_exterior_update_model=RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed"),
        )
    )
    corrected_connector_feedback_model.bind_state(state)

    shell_feedback = np.asarray(
        shell_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
    )
    connector_feedback = np.asarray(
        raw_connector_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
    )
    corrected_connector_feedback = np.asarray(
        corrected_connector_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
    )
    diff = shell_feedback - connector_feedback
    corrected_diff = shell_feedback - corrected_connector_feedback

    np.testing.assert_allclose(report["shell_feedback_norm"], np.linalg.norm(shell_feedback))
    np.testing.assert_allclose(
        report["connector_feedback_norm"], np.linalg.norm(connector_feedback)
    )
    np.testing.assert_allclose(report["difference_norm"], np.linalg.norm(diff))
    np.testing.assert_allclose(report["max_abs_difference"], np.max(np.abs(diff)))
    np.testing.assert_allclose(
        report["corrected_connector_feedback_norm"], np.linalg.norm(corrected_connector_feedback)
    )
    np.testing.assert_allclose(
        report["corrected_difference_norm"], np.linalg.norm(corrected_diff)
    )
    np.testing.assert_allclose(
        report["corrected_max_abs_difference"], np.max(np.abs(corrected_diff))
    )

    n = int(state.solution_space.index_length)
    psi_to_e = np.asarray(state.toroidal_to_E_coeffs.to_dense(), dtype=float)
    alpha_to_psi = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)
    rhs_from_e = np.asarray(shell_feedback_model.build_rhs_operator(tor), dtype=float)
    trace_lhs = np.asarray(tor.jr_to_psi_coeff_operator, dtype=float) @ (
        (1.0 / float(mu0)) * (rhs_from_e @ psi_to_e @ alpha_to_psi)
    )

    update_model = RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed")
    update_model.bind_state(state)
    trace_rhs = np.asarray(update_model.build_dtpsi_from_dtalpha_operator(tor), dtype=float)
    corrected_update_model = IncrementalIdealityCorrectedExteriorToroidalUpdateModel(
        shell_feedback_response_model=EquivalentNonlocalRadialShellResponseModel(),
        base_exterior_update_model=RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed"),
    )
    corrected_update_model.bind_state(state)
    corrected_trace_rhs = np.asarray(
        corrected_update_model.build_dtpsi_from_dtalpha_operator(tor), dtype=float
    )
    trace_diff = trace_lhs - trace_rhs
    corrected_trace_diff = trace_lhs - corrected_trace_rhs

    np.testing.assert_allclose(
        report["incremental_connector_trace_lhs_norm"], np.linalg.norm(trace_lhs)
    )
    np.testing.assert_allclose(
        report["incremental_connector_trace_rhs_norm"], np.linalg.norm(trace_rhs)
    )
    np.testing.assert_allclose(
        report["incremental_connector_trace_difference_norm"], np.linalg.norm(trace_diff)
    )
    np.testing.assert_allclose(
        report["incremental_connector_trace_max_abs_difference"], np.max(np.abs(trace_diff))
    )
    np.testing.assert_allclose(
        report["incremental_corrected_trace_rhs_norm"], np.linalg.norm(corrected_trace_rhs)
    )
    np.testing.assert_allclose(
        report["incremental_corrected_trace_difference_norm"],
        np.linalg.norm(corrected_trace_diff),
    )
    np.testing.assert_allclose(
        report["incremental_corrected_trace_max_abs_difference"],
        np.max(np.abs(corrected_trace_diff)),
    )


def test_radial_shell_forcing_comparison_report_matches_direct_operator_comparison() -> None:
    """Runtime report should reproduce the direct condensed-vs-explicit forcing comparison."""
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=10,
        Mmax=5,
        Ncs=12,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL.value,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        RM=10.0 * RE,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator=IntegratorKind.EULER,
        least_squares_solver="svd",
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    state = sim.state
    grid = state.geometry.grid
    sim.set_conductance(np.zeros(grid.size), np.ones(grid.size), lat=grid.lat, lon=grid.lon, time=None)
    state.update(sim.input_manager, sim.current_time, interpolation=True)

    report = state.get_radial_shell_forcing_comparison_report()
    tor = state.toroidal_matrices

    condensed_model = EquivalentNonlocalRadialShellResponseModel()
    condensed_model.bind_state(state)
    explicit_model = FrozenConductanceIncrementalKnownElectricRadialResponseModel()
    explicit_model.bind_state(state)

    condensed_rhs = np.asarray(condensed_model.build_rhs_operator(tor), dtype=float)
    explicit_rhs = np.asarray(explicit_model.build_rhs_operator(tor), dtype=float)
    diff = condensed_rhs - explicit_rhs

    np.testing.assert_allclose(report["condensed_operator_norm"], np.linalg.norm(condensed_rhs))
    np.testing.assert_allclose(report["explicit_operator_norm"], np.linalg.norm(explicit_rhs))
    np.testing.assert_allclose(report["difference_norm"], np.linalg.norm(diff))
    np.testing.assert_allclose(report["max_abs_difference"], np.max(np.abs(diff)))

    zero_e = np.zeros((2, state.solution_space.index_length), dtype=float).reshape(-1)
    explicit_action = explicit_rhs @ zero_e
    condensed_action = condensed_rhs @ zero_e
    np.testing.assert_allclose(
        report["component_report"]["total_external"]["explicit_action_norm"],
        np.linalg.norm(explicit_action),
    )
    np.testing.assert_allclose(
        report["component_report"]["total_external"]["condensed_action_norm"],
        np.linalg.norm(condensed_action),
    )
    assert "condensed_cf_action_norm" in report["component_report"]["total_external"]
    assert "explicit_df_action_norm" in report["component_report"]["total_external"]


def test_radial_shell_forcing_comparison_report_exposes_wind_mismatch() -> None:
    """Wind forcing should separate reduced inductive and incremental closures."""
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=10,
        Mmax=5,
        Ncs=12,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL.value,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        RM=10.0 * RE,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator=IntegratorKind.EULER,
        least_squares_solver="svd",
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    state = sim.state
    grid = state.geometry.grid
    sim.set_conductance(np.zeros(grid.size), np.ones(grid.size), lat=grid.lat, lon=grid.lon, time=None)
    sim.set_u(np.zeros(grid.size), 100.0 * np.ones(grid.size), lat=grid.lat, lon=grid.lon, time=None)
    state.update(sim.input_manager, sim.current_time, interpolation=True)

    report = state.get_radial_shell_forcing_comparison_report()
    wind = report["component_report"]["wind"]

    assert wind["condensed_action_norm"] < 1e-30
    assert wind["condensed_cf_action_norm"] < 1e-30
    assert wind["condensed_df_action_norm"] < 1e-30
    assert wind["explicit_action_norm"] > 0.0
    assert wind["difference_norm"] > 0.0
    assert wind["difference_to_explicit_ratio"] == pytest.approx(1.0)
    assert wind["explicit_cf_action_norm"] > 0.0
    assert wind["explicit_df_action_norm"] > 0.0


def test_poloidal_time_derivative_forcing_candidate_collapses_to_zero() -> None:
    """The ``E_df -> d m_ind/dt -> dJ_S/dt`` continuity candidate is null here."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=PFACNonlocalRadialShellResponseModel(),
    )
    grid = state.geometry.grid
    sim = type("SimLike", (), {})()
    # Reuse the live state by setting conductance directly through Dynamics-style APIs is not available here.
    # For this operator identity, unit Pedersen/Hall-free conductance is enough.
    from pynamit.primitives.field import Field

    ped = np.ones(grid.size, dtype=float)
    hall = np.zeros(grid.size, dtype=float)
    state.etaP = Field.from_grid_values(grid, ped, np.zeros(grid.size), np.zeros(grid.size))
    state.etaH = Field.from_grid_values(grid, hall, np.zeros(grid.size), np.zeros(grid.size))
    state._invalidate_caches()

    n = int(state.solution_space.index_length)
    extract_df = np.asarray(state.E_coeffs_to_E_df_matrix, dtype=float)
    scale = float(state.poloidal_matrices.E_df_to_d_m_ind_dt)
    m_ind_to_js = np.asarray(
        coerce_dense_operator_matrix(
            state.geometry.get_potential_to_JS_operator("m_ind", mode=None),
            n_component_rows=2,
            n_cols=n,
        ),
        dtype=float,
    )
    div_omega = np.asarray(
        coerce_dense_operator_matrix(state.solution_space.get_vector_divergence_operator(), n_cols=2 * n),
        dtype=float,
    )
    candidate = (-(float(mu0) / float(state.RI))) * (div_omega @ m_ind_to_js @ (scale * extract_df))

    np.testing.assert_allclose(candidate, 0.0, rtol=0.0, atol=1e-12)


def test_incremental_ideality_corrected_connector_matches_shell_feedback() -> None:
    """Corrected connector feedback should equal the shell-electric feedback operator."""
    equivalent_shell_model = EquivalentNonlocalRadialShellResponseModel()
    corrected_update = IncrementalIdealityCorrectedExteriorToroidalUpdateModel(
        shell_feedback_response_model=equivalent_shell_model,
        base_exterior_update_model=RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed"),
    )
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=NonlocalShellElectricRadialResponseModel(
            shell_response_model=equivalent_shell_model,
            exterior_update_model=corrected_update,
        ),
    )
    tor = state.toroidal_matrices

    shell_feedback_model = NonlocalShellElectricRadialResponseModel(
        shell_response_model=EquivalentNonlocalRadialShellResponseModel()
    )
    shell_feedback_model.bind_state(state)
    corrected_feedback_model = ExteriorToroidalScalarRadialResponseModel(
        exterior_update_model=corrected_update
    )
    corrected_feedback_model.bind_state(state)

    shell_feedback = np.asarray(
        shell_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
    )
    corrected_feedback = np.asarray(
        corrected_feedback_model.build_feedback_dtalpha_operator(tor), dtype=float
    )
    np.testing.assert_allclose(corrected_feedback, shell_feedback, rtol=1e-10, atol=1e-10)

    raw_update_model = RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed")
    raw_update_model.bind_state(state)
    base_dtpsi = np.asarray(raw_update_model.build_dtpsi_from_dtalpha_operator(tor), dtype=float)
    correction_dtpsi = np.asarray(
        corrected_update.build_correction_dtpsi_from_dtalpha_operator(tor), dtype=float
    )
    total_dtpsi = np.asarray(corrected_update.build_dtpsi_from_dtalpha_operator(tor), dtype=float)
    matched_dtpsi = np.asarray(
        corrected_update.build_matched_dtpsi_from_dtalpha_operator(tor), dtype=float
    )

    np.testing.assert_allclose(total_dtpsi, base_dtpsi + correction_dtpsi, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(total_dtpsi, matched_dtpsi, rtol=1e-12, atol=1e-12)


def test_rm_toroidal_boundary_update_model_open_matches_alpha_to_psi() -> None:
    """Open RM update should reduce to the direct shell ``dt_alpha -> dt_psi`` map."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, rm=10.0 * RE)
    tor = state.toroidal_matrices
    model = RMToroidalBoundaryUpdateModel(rm_boundary_mode="open")
    model.bind_state(state)

    actual = np.asarray(model.build_dtpsi_from_dtalpha_operator(tor), dtype=float)
    expected = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_rm_toroidal_boundary_update_model_closed_adds_internal_rm_contribution() -> None:
    """Closed RM update should add the internally continued RM boundary psi contribution."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, rm=10.0 * RE)
    tor = state.toroidal_matrices
    model = RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed")
    model.bind_state(state)

    actual = np.asarray(model.build_dtpsi_from_dtalpha_operator(tor), dtype=float)
    base = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)
    shift = np.asarray(
        to_dense(state.basis.get_radial_shift_operator(state.RM, state.RI, kind="internal")),
        dtype=float,
    )
    expected = base + shift @ np.asarray(state.toroidal_rm_boundary_operators.alpha_to_boundary_psi_rm, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_finite_rm_magnetic_boundary_bundle_is_consistent_with_existing_rm_operators() -> None:
    """Finite-``R_M`` magnetic bundle should expose the preferred outer magnetic pair."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, rm=10.0 * RE)
    ops = state.magnetic_rm_boundary_operators

    assert isinstance(ops, MagneticRMBoundaryOperators)
    np.testing.assert_allclose(
        np.asarray(ops.alpha_to_boundary_psi_rm, dtype=float),
        np.asarray(state.toroidal_rm_boundary_operators.alpha_to_boundary_psi_rm, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(ops.m_ind_to_br_rm_open, dtype=float),
        np.asarray(state.poloidal_rm_boundary_operators.m_ind_to_br_rm_open, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(
            ops.magnetic_potential_rm_to_br_rm @ ops.m_ind_to_magnetic_potential_rm_open,
            dtype=float,
        ),
        np.asarray(ops.m_ind_to_br_rm_open, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(
            ops.magnetic_potential_rm_to_br_rm @ ops.m_ind_to_magnetic_potential_rm_effective,
            dtype=float,
        ),
        np.asarray(ops.m_ind_to_br_rm_effective, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_finite_rm_magnetic_boundary_bundle_vanishes_without_rm() -> None:
    """Without ``R_M`` the explicit outer magnetic boundary pair should be zero."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, rm=None)
    ops = state.magnetic_rm_boundary_operators

    assert isinstance(ops, MagneticRMBoundaryOperators)
    for arr in (
        ops.alpha_to_boundary_psi_rm,
        ops.magnetic_potential_rm_to_br_rm,
        ops.br_rm_to_magnetic_potential_rm,
        ops.m_ind_to_br_rm_open,
        ops.m_ind_to_br_rm_effective,
        ops.m_ind_to_br_rm_shielding,
        ops.m_ind_to_magnetic_potential_rm_open,
        ops.m_ind_to_magnetic_potential_rm_effective,
        ops.m_ind_to_magnetic_potential_rm_shielding,
    ):
        np.testing.assert_allclose(np.asarray(arr, dtype=float), 0.0, rtol=0.0, atol=0.0)


def test_trace_forcing_with_rm_update_feedback_matches_explicit_feedback_adapter() -> None:
    """Hybrid radial-shell model should use explicit RM toroidal update for feedback."""
    update_model = RMToroidalBoundaryUpdateModel(rm_boundary_mode="closed")
    hybrid_model = NonlocalShellElectricRadialResponseModel(
        shell_trace_model=HarmonicShellElectricTraceModel(outer_boundary_mode="open"),
        exterior_update_model=update_model,
    )
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        rm=10.0 * RE,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=hybrid_model,
    )
    tor = state.toroidal_matrices

    adapter = ExteriorToroidalScalarRadialResponseModel(exterior_update_model=update_model)
    adapter.bind_state(state)
    expected = np.asarray(adapter.build_feedback_dtalpha_operator(tor), dtype=float)
    actual = np.asarray(tor.full_radial_shell_feedback_dtalpha_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_nonlocal_shell_electric_radial_shell_lhs_matches_explicit_composition() -> None:
    """Explicit shell-electric radial closure should assemble ``mass - feedback``."""
    model = NonlocalShellElectricRadialResponseModel()
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=model,
    )
    tor = state.toroidal_matrices

    expected = (
        np.asarray(tor.radial_shell_mass_dtalpha_operator, dtype=float)
        - np.asarray(tor.full_radial_shell_feedback_dtalpha_operator, dtype=float)
    )
    actual = np.asarray(tor.dtalpha_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_harmonic_shell_trace_model_uses_repo_cf_sign_for_u() -> None:
    """The harmonic trace model should use ``U = cf_sign * Phi`` on the shell."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    trace_model = HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    trace_op = np.asarray(trace_model.build_trace_operator(tor), dtype=float)

    degrees = np.asarray(state.basis.n, dtype=float).reshape(-1)
    expected_phi_to_dudr = np.diag(float(get_repo_cf_helmholtz_sign()) * degrees / float(state.RI))

    np.testing.assert_allclose(trace_op[:n, :n], expected_phi_to_dudr, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(trace_op[:n, n:], 0.0, rtol=1e-12, atol=1e-12)


def test_q_trace_builder_uses_repo_cf_sign_for_physical_u(monkeypatch) -> None:
    """The ``q`` trace should be built from the physical ``U = cf_sign * Phi`` scalar."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    monkeypatch.setattr(
        radial_shell_response_module,
        "_build_ideal_er_from_shell_e_operator",
        lambda toroidal_matrices: np.zeros((n, 2 * n), dtype=float),
    )

    q_trace_op = np.asarray(
        build_q_trace_operator_from_poloidal_side_trace(tor, _IdentityPoloidalSideTraceModel()),
        dtype=float,
    )

    expected_phi_block = float(get_repo_cf_helmholtz_sign()) * np.eye(n, dtype=float)

    np.testing.assert_allclose(q_trace_op[:, :n], expected_phi_block, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q_trace_op[:, n:], 0.0, rtol=1e-12, atol=1e-12)


def test_harmonic_poloidal_side_trace_model_matches_open_dtn() -> None:
    """The direct harmonic side operator should be the open DtN map ``U -> d_r U``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    side_model = HarmonicPoloidalSideTraceModel(outer_boundary_mode="open")
    dudr_from_u = np.asarray(side_model.build_dudr_from_u_operator(tor), dtype=float)

    degrees = np.asarray(state.basis.n, dtype=float).reshape(-1)
    expected = np.diag(degrees / float(state.RI))

    np.testing.assert_allclose(dudr_from_u, expected, rtol=1e-12, atol=1e-12)


def test_harmonic_shell_trace_model_does_not_continue_toroidal_electric_channel() -> None:
    """The harmonic continuation should act only on the poloidal electric scalar."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    trace_model = HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    trace_op = np.asarray(trace_model.build_trace_operator(tor), dtype=float)

    np.testing.assert_allclose(trace_op[:n, n:], 0.0, rtol=1e-12, atol=1e-12)


def test_trace_based_radial_shell_rhs_matches_trace_assembly() -> None:
    """Trace-based radial-shell rhs should assemble as ``Delta_S (E_r - d_r U)``."""
    response_model = NonlocalShellElectricRadialResponseModel(
        shell_trace_model=HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    )
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=response_model,
    )
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    trace_op = np.asarray(response_model.shell_trace_model.build_trace_operator(tor), dtype=float)
    lap = np.asarray(to_dense(state.basis.get_laplacian_operator(r=state.RI)), dtype=float)
    expected = lap @ (trace_op[n:] - trace_op[:n])
    actual = np.asarray(tor.full_radial_shell_rhs_from_E_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_trace_based_nonlocal_shell_electric_radial_lhs_matches_explicit_composition() -> None:
    """Trace-based shell-electric radial closure should assemble ``mass - feedback``."""
    model = NonlocalShellElectricRadialResponseModel(
        shell_trace_model=HarmonicShellElectricTraceModel(outer_boundary_mode="open")
    )
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        toroidal_closure_mode="radial_shell",
        radial_shell_response_model=model,
    )
    tor = state.toroidal_matrices

    expected = (
        np.asarray(tor.radial_shell_mass_dtalpha_operator, dtype=float)
        - np.asarray(tor.full_radial_shell_feedback_dtalpha_operator, dtype=float)
    )
    actual = np.asarray(tor.dtalpha_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_state_dpsi_solver_accepts_direct_e_forcing() -> None:
    """State-level toroidal solve remains operational with direct E forcing only."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    n = int(state.solution_space.index_length)
    rng = np.random.default_rng(37)
    E_known = rng.normal(size=(2, n))
    dt_psi = np.asarray(state.solve_dt_psi(E_known), dtype=float).reshape(-1)
    assert dt_psi.size == n
    assert np.all(np.isfinite(dt_psi))


def test_coupled_dt_psi_from_m_ind_matches_direct_e_chain() -> None:
    """Coupled ``dt_psi_from_m_ind`` block is exactly the direct ``m_ind -> E -> dt_psi`` chain."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    blocks = state.get_coupled_induction_blocks(source="dense")
    api = state.coupled_operators
    dt_psi_from_E = np.asarray(
        api._get_dt_psi_from_E_dense(apply_psi_gauge=state.apply_psi_gauge), dtype=float
    )
    m_ind_to_E = np.asarray(
        api._dense_E_coeff_operator_matrix(state.m_ind_to_E_coeffs), dtype=float
    )
    expected = dt_psi_from_E @ m_ind_to_E

    assert np.allclose(
        np.asarray(blocks["dt_psi_from_m_ind"], dtype=float), expected, rtol=1e-10, atol=1e-10
    )


def test_coupled_dense_sparse_parity() -> None:
    """Coupled dense and sparse assembly remain consistent."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    dense = np.asarray(
        state.get_coupled_induction_matrix(source="dense", flatten=True), dtype=float
    )
    sparse = np.asarray(
        state.get_coupled_induction_matrix(source="sparse", flatten=True), dtype=float
    )
    assert np.allclose(dense, sparse, rtol=1e-9, atol=1e-9)


def test_wind_cross_product_tensor_matches_minus_u_cross_b() -> None:
    """Geometry ``bu`` should implement ``E = -u x B = B x u`` for horizontal winds."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)

    bu = np.asarray(state.geometry.bu, dtype=float)
    br = np.asarray(state.geometry.b_field.vec.r, dtype=float).reshape(-1)
    rng = np.random.default_rng(19)
    u_grid = rng.normal(size=(2, br.size))

    actual = np.einsum("abg,bg->ag", bu, u_grid, optimize=True)
    expected = np.vstack((-br * u_grid[1], br * u_grid[0]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_poloidal_grid_js_sign_identities_match_operator_definitions() -> None:
    """Grid-level JS operators should match the intended ``m_ind`` and PFAC sign chain."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    mats = state.poloidal_matrices

    grad = np.asarray(
        to_dense(state.solution_space.get_gradient_matrix(state.geometry.grid)), dtype=float
    ).reshape(2, -1, state.solution_space.index_length)
    curl = np.asarray(
        to_dense(state.solution_space.get_curl_matrix(state.geometry.grid)), dtype=float
    ).reshape(2, -1, state.solution_space.index_length)
    scaling = np.asarray(
        to_dense(state.solution_space.get_potential_scaling_operator()), dtype=float
    )

    expected_g_ve = np.tensordot(
        (-1.0 / (mu0 * float(get_repo_df_helmholtz_sign()))) * curl,
        scaling,
        axes=([2], [0]),
    )
    expected_g_mimp_local = (-1.0 / mu0) * grad
    expected_g_mimp = expected_g_mimp_local + np.tensordot(
        expected_g_ve,
        np.asarray(mats._apply_imposed_toroidal_shielding(mats.T_to_Ve), dtype=float),
        axes=([2], [0]),
    )

    np.testing.assert_allclose(np.asarray(mats.G_Ve_to_JS), expected_g_ve, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(mats.G_m_ind_to_JS), expected_g_ve, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(mats.G_m_imp_to_JS), expected_g_mimp, rtol=2e-12, atol=1e-12
    )


def test_sh_advection_and_psi_scalings_match_closed_form() -> None:
    """Check raw advection and jr->psi scalings against direct SH assembly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    basis = state.basis
    grid = state.geometry.grid

    advection_raw = np.asarray(tor.fieldline_advection_operator_raw, dtype=float)
    jr_to_psi = np.asarray(tor.jr_to_psi_coeff_operator, dtype=float)
    G = np.asarray(basis.get_evaluation_matrix(grid), dtype=float)
    G_th = np.asarray(basis.get_evaluation_matrix(grid, derivative="theta"), dtype=float)
    G_ph = np.asarray(basis.get_evaluation_matrix(grid, derivative="phi"), dtype=float)
    weights = np.asarray(grid.weights, dtype=float).reshape(-1)

    Bth = np.asarray(state.geometry.b_field.vec.theta, dtype=float).reshape(-1)
    Bph = np.asarray(state.geometry.b_field.vec.phi, dtype=float).reshape(-1)
    A = (G.T * weights) @ ((Bth[:, None] * G_th) + (Bph[:, None] * G_ph))

    l = np.asarray(basis.n, dtype=float).reshape(-1)
    laplacian_eigenvalues = l * (l + 1.0)
    mask = laplacian_eigenvalues > 0
    inverse_laplacian_eigenvalues = np.zeros_like(laplacian_eigenvalues)
    inverse_laplacian_eigenvalues[mask] = 1.0 / laplacian_eigenvalues[mask]

    jr_to_psi_ref = np.diag(mu0 * float(state.RI) * inverse_laplacian_eigenvalues)

    assert np.allclose(advection_raw, A, rtol=1e-10, atol=1e-10)
    assert np.allclose(jr_to_psi[:, mask], jr_to_psi_ref[:, mask], rtol=1e-10, atol=1e-10)
    assert np.linalg.norm(jr_to_psi[:, ~mask]) < 1e-12


def test_jr_to_psi_matches_m_imp_inverse_sign_convention() -> None:
    """``psi`` and ``m_imp`` should invert ``jr`` with the same sign convention."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    m_imp_to_jr = np.asarray(to_dense(state.poloidal_matrices.m_imp_to_jr), dtype=float)
    jr_to_psi = np.asarray(tor.jr_to_psi_coeff_operator, dtype=float)
    l = np.asarray(state.basis.n, dtype=float).reshape(-1)
    mask = (l * (l + 1.0)) > 0.0
    projector = np.diag(mask.astype(float))

    assert np.allclose(jr_to_psi @ m_imp_to_jr, projector, rtol=1e-10, atol=1e-10)
    assert np.allclose(m_imp_to_jr @ jr_to_psi, projector, rtol=1e-10, atol=1e-10)


def test_m_imp_modal_scaling_is_negated_relative_to_paper_t_coefficients() -> None:
    """Repo ``m_imp`` uses the opposite SH sign from the paper's ``psi,xi`` family."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)

    m_imp_to_jr = np.asarray(to_dense(state.poloidal_matrices.m_imp_to_jr), dtype=float)
    n = np.asarray(state.basis.n, dtype=float).reshape(-1)

    # Repo convention: jr = -(R/mu0) Delta_S(m_imp)
    # -> jr_nm = +(n(n+1)/(mu0 R)) * m_imp_nm
    expected_repo = np.diag((n * (n + 1.0)) / (mu0 * float(state.RI)))
    np.testing.assert_allclose(m_imp_to_jr, expected_repo, rtol=1e-12, atol=1e-12)

    # Paper convention (Eq. 26): jr_nm = -(n(n+1)/(mu0 R)) * T_nm
    expected_paper = -expected_repo
    assert np.linalg.norm(m_imp_to_jr - expected_paper) > 1e-6 * max(
        np.linalg.norm(expected_paper), 1.0
    )


def test_dtalpha_feedback_psi_rewrite_matches_dtjr_form() -> None:
    """Psi rewrite of the feedback block must equal the direct dt_alpha closed form."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    advection_raw = np.asarray(tor.fieldline_advection_operator_raw, dtype=float)
    jr_to_psi = np.asarray(tor.jr_to_psi_coeff_operator, dtype=float)
    alpha_to_jr = np.asarray(tor.alpha_to_jr_coeff_operator, dtype=float)
    radial_closure_dtalpha = np.asarray(tor.radial_closure_dtalpha, dtype=float)
    inv_R = 1.0 / float(state.RI)
    feedback_ref = advection_raw @ (
        (2.0 * inv_R) * (jr_to_psi @ alpha_to_jr) + (jr_to_psi @ radial_closure_dtalpha)
    )

    alpha_to_psi = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)
    radial_closure_dtpsi = np.asarray(tor.radial_closure_dt_psi_from_dtalpha, dtype=float)
    feedback_psi = advection_raw @ ((inv_R * alpha_to_psi) + radial_closure_dtpsi)

    assert np.allclose(feedback_psi, feedback_ref, rtol=1e-10, atol=1e-10)
    assert np.allclose(
        np.asarray(tor.toroidal_potential_feedback_dtalpha_operator, dtype=float),
        feedback_ref,
        rtol=1e-10,
        atol=1e-10,
    )


def test_first_principles_projected_dtalpha_operator_matches_mass_minus_feedback() -> None:
    """First-principles projected tangential operator should be ``mass - feedback``."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices

    expected = np.asarray(tor.mass_dtalpha, dtype=float) - np.asarray(
        tor.toroidal_potential_feedback_dtalpha_operator, dtype=float
    )
    actual = np.asarray(tor.first_principles_projected_dtalpha_operator, dtype=float)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_projected_closure_report_exposes_perpendicular_component() -> None:
    """Projected-closure diagnostics should report the dropped perpendicular activity."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)

    report = state.get_toroidal_projected_closure_report()

    assert "operator_report" in report
    assert "component_report" in report
    assert report["operator_report"]["first_principles_projected_norm"] > 0.0
    assert report["operator_report"]["current_minus_projected_norm"] >= 0.0
    assert "total_external" in report["component_report"]
    assert report["component_report"]["total_external"]["perpendicular_action_norm"] >= 0.0


def test_internal_tangential_shadow_stacks_projected_and_perpendicular_blocks() -> None:
    """The internal tangential benchmark shadow should retain both shell-balance components."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    tor = _build_shadow_tangential_toroidal_matrices(state)
    n = int(state.solution_space.index_length)

    rhs_op = np.asarray(tor.toroidal_rhs_from_E_operator, dtype=float)
    physics_op = np.asarray(tor.physics_residual_coeff_operator, dtype=float)
    dtalpha_from_rhs = np.asarray(
        tor.build_dtalpha_from_toroidal_rhs_matrix(
            constraint_operator=state.dt_alpha_constraint_system.hard_operator,
            weighting=state.toroidal_weighting,
            regularization_lambda=state.toroidal_regularization_lambda,
            penalty_operator=state.dt_alpha_constraint_system.soft_operator,
            penalty_scaling=float(state.dt_alpha_constraint_system.soft_scaling),
            hinv_rtol=0.0,
        ),
        dtype=float,
    )

    assert rhs_op.shape == (2 * n, 2 * n)
    assert physics_op.shape == (2 * n, n)
    assert dtalpha_from_rhs.shape == (n, 2 * n)

    np.testing.assert_allclose(
        physics_op[:n],
        np.asarray(tor.first_principles_projected_dtalpha_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        physics_op[n:],
        np.asarray(tor.first_principles_perpendicular_dtalpha_operator, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_internal_tangential_shadow_direct_e_map_runs() -> None:
    """The internal tangential benchmark shadow should still produce finite direct-E maps."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    tor = _build_shadow_tangential_toroidal_matrices(state)
    n = int(state.solution_space.index_length)
    rng = np.random.default_rng(41)
    E_known = rng.normal(size=2 * n)
    dtalpha_from_rhs = np.asarray(
        tor.build_dtalpha_from_toroidal_rhs_matrix(
            constraint_operator=state.dt_alpha_constraint_system.hard_operator,
            weighting=state.toroidal_weighting,
            regularization_lambda=state.toroidal_regularization_lambda,
            penalty_operator=state.dt_alpha_constraint_system.soft_operator,
            penalty_scaling=float(state.dt_alpha_constraint_system.soft_scaling),
            hinv_rtol=0.0,
        ),
        dtype=float,
    )
    alpha_to_psi = np.asarray(tor.alpha_to_psi_coeff_operator, dtype=float)
    dt_psi = np.asarray(
        alpha_to_psi @ dtalpha_from_rhs @ np.asarray(tor.toroidal_rhs_from_E_operator, dtype=float) @ E_known,
        dtype=float,
    ).reshape(-1)
    assert dt_psi.shape == (n,)
    assert np.all(np.isfinite(dt_psi))


def test_wind_and_jr_forcing_superpose_linearly_in_dt_psi() -> None:
    """Wind-driven and ``jr``-driven toroidal forcing should remain odd and linear."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=8, mmax=4)
    n = int(state.solution_space.index_length)
    rng = np.random.default_rng(23)

    u_coeffs = rng.normal(size=(2, n))
    jr_coeffs = rng.normal(size=n)

    e_u = np.asarray(
        state._apply_operator(state.u_coeffs_to_E_coeffs, u_coeffs, (2, n)), dtype=float
    )
    m_imp_from_jr = np.asarray(
        state.get_m_imp_from_jr_matrix(input_basis=state.solution_space), dtype=float
    )
    m_imp = m_imp_from_jr @ jr_coeffs
    e_jr = np.asarray(
        state._apply_operator(state.m_imp_to_E_coeffs, m_imp, (2, n)), dtype=float
    )

    dt_psi_u = np.asarray(state.solve_dt_psi(e_u), dtype=float)
    dt_psi_jr = np.asarray(state.solve_dt_psi(e_jr), dtype=float)
    dt_psi_sum = np.asarray(state.solve_dt_psi(e_u + e_jr), dtype=float)
    dt_psi_neg_u = np.asarray(state.solve_dt_psi(-e_u), dtype=float)

    np.testing.assert_allclose(dt_psi_neg_u, -dt_psi_u, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dt_psi_sum, dt_psi_u + dt_psi_jr, rtol=1e-10, atol=1e-10)


def test_mass_dtalpha_matrix_is_symmetric() -> None:
    """`mass_dtalpha` should remain symmetric after |B_s|^2 factor change."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    mass_dtalpha_matrix = np.asarray(state.toroidal_matrices.mass_dtalpha, dtype=float)
    asym = mass_dtalpha_matrix - mass_dtalpha_matrix.T
    rel = np.linalg.norm(asym) / max(np.linalg.norm(mass_dtalpha_matrix), 1e-30)
    assert rel < 1e-10


def test_cs_div_rxcurl_identity() -> None:
    """Discrete identity: div(rhat×a) = -curl(a) on CS derivative operators."""
    state = _build_state(simulation_mode=SimulationMode.CS_DOMINANT, nmax=8, mmax=4, ncs=12)
    tor = state.toroidal_matrices
    D_th, D_ph, _ = [np.asarray(x, dtype=float) for x in tor.cs_grid_derivative_operators]
    theta = np.deg2rad(np.asarray(state.geometry.grid.theta, dtype=float).reshape(-1))
    sin_th = np.sin(theta)
    sin_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
    cot = np.cos(theta) / sin_safe

    rng = np.random.default_rng(11)
    a_th = rng.normal(size=theta.size)
    a_ph = rng.normal(size=theta.size)

    v_th = -a_ph
    v_ph = a_th
    div_v = (D_th @ v_th) + cot * v_th + (D_ph @ v_ph)
    curl_a = (D_th @ a_ph) + cot * a_ph - (D_ph @ a_th)

    rel = np.linalg.norm(div_v + curl_a) / max(np.linalg.norm(curl_a), 1e-30)
    assert rel < 1e-10
