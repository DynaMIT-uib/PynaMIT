"""Targeted toroidal operator checks for the LaTeX-consistent formulation."""

from __future__ import annotations

import numpy as np

from pynamit.math.constants import mu0
from pynamit.simulation.spatial import to_dense
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import SimulationMode


def _build_state(
    *,
    simulation_mode: SimulationMode = SimulationMode.PURE_SPECTRAL,
    nmax: int = 10,
    mmax: int = 5,
    ncs: int = 12,
):
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=nmax,
        Mmax=mmax,
        Ncs=ncs,
        dynamics_mode="full_induction",
        simulation_mode=simulation_mode.value,
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


def test_toroidal_forcing_gradient_field_is_zero() -> None:
    """Er-free forcing must vanish for pure cf (gradient) tangential E."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)

    rng = np.random.default_rng(7)
    phi = rng.normal(size=n)
    E_coeffs = np.vstack([phi, np.zeros_like(phi)])

    forcing = np.asarray(tor.compute_toroidal_rhs_from_E(E_coeffs), dtype=float).reshape(-1)
    assert np.linalg.norm(forcing) < 1e-10


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
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        nmax=8,
        mmax=4,
    )
    blocks = state.get_coupled_induction_blocks(source="dense", use_pinning=state.apply_psi_gauge)
    api = state.coupled_operators
    dt_psi_from_E = np.asarray(
        api._get_dt_psi_from_E_dense(use_pinning=state.apply_psi_gauge),
        dtype=float,
    )
    m_ind_to_E = np.asarray(api._dense_E_coeff_operator_matrix(state.m_ind_to_E_coeffs), dtype=float)
    expected = dt_psi_from_E @ m_ind_to_E

    assert np.allclose(np.asarray(blocks["dt_psi_from_m_ind"], dtype=float), expected, rtol=1e-10, atol=1e-10)


def test_coupled_dense_sparse_parity() -> None:
    """Coupled dense and sparse assembly remain consistent."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        nmax=8,
        mmax=4,
    )
    dense = np.asarray(
        state.get_coupled_induction_matrix(source="dense", flatten=True, use_pinning=state.apply_psi_gauge),
        dtype=float,
    )
    sparse = np.asarray(
        state.get_coupled_induction_matrix(source="sparse", flatten=True, use_pinning=state.apply_psi_gauge),
        dtype=float,
    )
    assert np.allclose(dense, sparse, rtol=1e-9, atol=1e-9)


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

    jr_to_psi_ref = np.diag(-mu0 * float(state.RI) * inverse_laplacian_eigenvalues)

    assert np.allclose(advection_raw, A, rtol=1e-10, atol=1e-10)
    assert np.allclose(jr_to_psi[:, mask], jr_to_psi_ref[:, mask], rtol=1e-10, atol=1e-10)
    assert np.linalg.norm(jr_to_psi[:, ~mask]) < 1e-12


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


def test_mass_dtalpha_matrix_is_symmetric() -> None:
    """`mass_dtalpha` should remain symmetric after |B_s|^2 factor change."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    mass_dtalpha_matrix = np.asarray(state.toroidal_matrices.mass_dtalpha, dtype=float)
    asym = mass_dtalpha_matrix - mass_dtalpha_matrix.T
    rel = np.linalg.norm(asym) / max(np.linalg.norm(mass_dtalpha_matrix), 1e-30)
    assert rel < 1e-10


def test_cs_div_rxcurl_identity() -> None:
    """Discrete identity: div(rhat×a) = -curl(a) on CS derivative operators."""
    state = _build_state(
        simulation_mode=SimulationMode.CS_DOMINANT,
        nmax=8,
        mmax=4,
        ncs=12,
    )
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
