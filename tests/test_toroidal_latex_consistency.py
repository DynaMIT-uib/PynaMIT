"""Targeted toroidal operator checks for the LaTeX-consistent formulation."""

from __future__ import annotations

import numpy as np
import pytest

from pynamit.math.constants import mu0
from pynamit.simulation.geometry_utils import to_dense
from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import SimulationMode


def _build_state(
    *,
    simulation_mode: SimulationMode = SimulationMode.PURE_SPECTRAL,
    nmax: int = 10,
    mmax: int = 5,
    ncs: int = 12,
    use_toroidal_u_known_from_poloidal: bool = False,
    toroidal_u_known_radial_model: str = "none",
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
        use_toroidal_u_known_from_poloidal=use_toroidal_u_known_from_poloidal,
        toroidal_u_known_radial_model=toroidal_u_known_radial_model,
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

    forcing = np.asarray(tor.compute_toroidal_forcing_from_E(E_coeffs), dtype=float).reshape(-1)
    assert np.linalg.norm(forcing) < 1e-10


def test_toroidal_forcing_u_known_zero_reduces_to_e_only() -> None:
    """Providing zero known-u inputs must match the E-only forcing path."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    n_grid = int(np.asarray(state.geometry.grid.theta, dtype=float).reshape(-1).size)

    rng = np.random.default_rng(19)
    E_coeffs = rng.normal(size=(2, n))
    u_zero = np.zeros((2, n_grid), dtype=float)

    forcing_ref = np.asarray(tor.compute_toroidal_forcing_from_E(E_coeffs), dtype=float).reshape(-1)
    forcing_u = np.asarray(
        tor.compute_toroidal_forcing_from_E(
            E_coeffs,
            u_known_grid=u_zero,
            dr_u_known_grid=u_zero,
        ),
        dtype=float,
    ).reshape(-1)
    assert np.allclose(forcing_u, forcing_ref, rtol=1e-12, atol=1e-12)


def test_toroidal_forcing_u_known_requires_dr_u_known() -> None:
    """Known-u forcing must provide dr_u unless explicit opt-in is enabled."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL)
    tor = state.toroidal_matrices
    n = int(state.basis.index_length)
    n_grid = int(np.asarray(state.geometry.grid.theta, dtype=float).reshape(-1).size)

    E_coeffs = np.zeros((2, n), dtype=float)
    u_only = np.zeros((2, n_grid), dtype=float)
    with pytest.raises(ValueError, match="dr_u_known_grid"):
        _ = tor.compute_toroidal_forcing_from_E(E_coeffs, u_known_grid=u_only)


def test_sh_u_known_forcing_matches_manual_grid_formula() -> None:
    """SH known-u forcing matches direct grid-formula assembly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    basis = state.basis
    grid = state.geometry.grid

    P = np.asarray(to_dense(tor.projection_matrix), dtype=float)
    G_th = np.asarray(to_dense(basis.get_evaluation_matrix(grid, derivative="theta")), dtype=float)
    G_ph = np.asarray(to_dense(basis.get_evaluation_matrix(grid, derivative="phi")), dtype=float)

    Br = np.asarray(state.geometry.b_field.vec.r, dtype=float).reshape(-1)
    Bth = np.asarray(state.geometry.b_field.vec.theta, dtype=float).reshape(-1)
    Bph = np.asarray(state.geometry.b_field.vec.phi, dtype=float).reshape(-1)
    theta = np.deg2rad(np.asarray(grid.theta, dtype=float).reshape(-1))
    sin_th = np.sin(theta)
    sin_safe = np.where(np.abs(sin_th) < 1e-12, 1e-12, sin_th)
    cot = np.cos(theta) / sin_safe
    inv_Rb = 1.0 / float(state.RI)

    rng = np.random.default_rng(23)
    u_th = rng.normal(size=theta.size)
    u_ph = rng.normal(size=theta.size)
    dr_u_th = rng.normal(size=theta.size)
    dr_u_ph = rng.normal(size=theta.size)

    div_u = (G_th @ (P @ u_th)) + cot * u_th + (G_ph @ (P @ u_ph))
    S_u = (
        (Br * inv_Rb) * div_u
        + Bth * (-inv_Rb * u_th - dr_u_th)
        + Bph * (-inv_Rb * u_ph - dr_u_ph)
    )
    forcing_ref = P @ S_u

    E_zero = np.zeros((2, int(basis.index_length)), dtype=float)
    forcing = np.asarray(
        tor.compute_toroidal_forcing_from_E(
            E_zero,
            u_known_grid=(u_th, u_ph),
            dr_u_known_grid=(dr_u_th, dr_u_ph),
        ),
        dtype=float,
    ).reshape(-1)

    assert np.allclose(forcing, forcing_ref, rtol=1e-10, atol=1e-10)


def test_poloidal_u_known_builder_matches_manual_none_model() -> None:
    """Poloidal provider returns the expected u/dr_u for radial_model='none'."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    pol = state.poloidal_matrices
    basis = state.solution_basis
    grid = state.geometry.grid
    n = int(basis.index_length)
    inv_R = 1.0 / float(state.RI)

    G_th = np.asarray(to_dense(basis.get_evaluation_matrix(grid, derivative="theta")), dtype=float)
    G_ph = np.asarray(to_dense(basis.get_evaluation_matrix(grid, derivative="phi")), dtype=float)

    rng = np.random.default_rng(29)
    dm = rng.normal(size=n)
    u_known, dr_u_known = pol.build_toroidal_u_known_terms_from_dm_ind_dt(dm, radial_model="none")

    u_theta_ref = inv_R * (G_ph @ dm)
    u_phi_ref = -inv_R * (G_th @ dm)
    u_ref = np.vstack([u_theta_ref, u_phi_ref])

    assert np.allclose(u_known, u_ref, rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(dr_u_known) < 1e-12


def test_poloidal_u_known_builder_external_model_scales_dr_u() -> None:
    """External model dr_u follows mode-wise -(l+2)/R scaling."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    pol = state.poloidal_matrices
    basis = state.solution_basis
    grid = state.geometry.grid
    n = int(basis.index_length)
    inv_R = 1.0 / float(state.RI)

    G_th = np.asarray(to_dense(basis.get_evaluation_matrix(grid, derivative="theta")), dtype=float)
    G_ph = np.asarray(to_dense(basis.get_evaluation_matrix(grid, derivative="phi")), dtype=float)
    l_arr = np.asarray(basis.n, dtype=float).reshape(-1)

    rng = np.random.default_rng(31)
    dm = rng.normal(size=n)
    u_known, dr_u_known = pol.build_toroidal_u_known_terms_from_dm_ind_dt(
        dm, radial_model="external_lplus2"
    )

    beta = -((l_arr + 2.0) / float(state.RI))
    dm_beta = beta * dm
    dr_u_theta_ref = inv_R * (G_ph @ dm_beta)
    dr_u_phi_ref = -inv_R * (G_th @ dm_beta)
    dr_u_ref = np.vstack([dr_u_theta_ref, dr_u_phi_ref])

    assert np.allclose(dr_u_known, dr_u_ref, rtol=1e-12, atol=1e-12)
    # Sanity: u itself is unchanged by radial model choice.
    u_none, _ = pol.build_toroidal_u_known_terms_from_dm_ind_dt(dm, radial_model="none")
    assert np.allclose(u_known, u_none, rtol=1e-12, atol=1e-12)


def test_state_dpsi_solver_accepts_poloidal_u_known_hook() -> None:
    """State-level hook wiring for optional poloidal u-known forcing is operational."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        use_toroidal_u_known_from_poloidal=True,
        toroidal_u_known_radial_model="none",
    )
    n = int(state.solution_basis.index_length)
    rng = np.random.default_rng(37)
    E_known = rng.normal(size=(2, n))
    dpsi = np.asarray(state.solve_dpsi_dt(E_known), dtype=float).reshape(-1)
    assert dpsi.size == n
    assert np.all(np.isfinite(dpsi))


def test_coupled_dtpsi_from_mind_includes_poloidal_u_known_hook() -> None:
    """Coupled ``dtpsi_from_mind`` block includes optional known-u correction."""
    state_off = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        nmax=8,
        mmax=4,
        use_toroidal_u_known_from_poloidal=False,
    )
    state_on = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        nmax=8,
        mmax=4,
        use_toroidal_u_known_from_poloidal=True,
        toroidal_u_known_radial_model="none",
    )

    blocks_off = state_off.get_coupled_induction_blocks(source="dense", use_pinning=state_off.apply_psi_gauge)
    blocks_on = state_on.get_coupled_induction_blocks(source="dense", use_pinning=state_on.apply_psi_gauge)
    delta = np.asarray(blocks_on["dtpsi_from_mind"], dtype=float) - np.asarray(
        blocks_off["dtpsi_from_mind"], dtype=float
    )

    api_on = state_on.coupled_operator_api
    dpsi_from_forcing = np.asarray(
        api_on._get_dtpsi_from_forcing_dense(use_pinning=state_on.apply_psi_gauge), dtype=float
    )
    forcing_from_dmindt = np.asarray(api_on._get_u_known_forcing_from_dmindt_dense(), dtype=float)
    dmind_from_mind = np.asarray(blocks_on["dmind_from_mind"], dtype=float)
    expected = dpsi_from_forcing @ forcing_from_dmindt @ dmind_from_mind

    assert np.linalg.norm(expected) > 0.0
    assert np.allclose(delta, expected, rtol=1e-10, atol=1e-10)


def test_coupled_u_known_hook_dense_sparse_parity() -> None:
    """Coupled dense and sparse assembly remain consistent with u-known hook enabled."""
    state = _build_state(
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        nmax=8,
        mmax=4,
        use_toroidal_u_known_from_poloidal=True,
        toroidal_u_known_radial_model="none",
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


def test_sh_coupling_matrix_scalings_match_closed_form() -> None:
    """Check coupling block scalings against direct SH quadrature assembly."""
    state = _build_state(simulation_mode=SimulationMode.PURE_SPECTRAL, nmax=10, mmax=5)
    tor = state.toroidal_matrices
    basis = state.basis
    grid = state.geometry.grid

    mapping_dtjr, mapping_dr_dtjr = [np.asarray(x, dtype=float) for x in tor.fieldline_mapping_dtjr_blocks]
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

    mapping_dtjr_ref = A * ((-2.0 * mu0 * inverse_laplacian_eigenvalues)[None, :])
    mapping_dr_dtjr_ref = A * (
        (-mu0 * float(state.RI) * inverse_laplacian_eigenvalues)[None, :]
    )

    assert np.allclose(mapping_dtjr[:, mask], mapping_dtjr_ref[:, mask], rtol=1e-10, atol=1e-10)
    assert np.allclose(
        mapping_dr_dtjr[:, mask], mapping_dr_dtjr_ref[:, mask], rtol=1e-10, atol=1e-10
    )
    assert np.linalg.norm(mapping_dtjr[:, ~mask]) < 1e-12
    assert np.linalg.norm(mapping_dr_dtjr[:, ~mask]) < 1e-12


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
