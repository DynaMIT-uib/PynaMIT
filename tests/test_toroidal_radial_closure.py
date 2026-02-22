"""Regression tests for stripped radial/isotropic toroidal closure behavior."""

from __future__ import annotations

import numpy as np

from pynamit.simulation.dynamics import Dynamics, SimulationMode


def _build_radial_isotropic_state():
    dynamics = Dynamics(
        filename_prefix="test_toroidal_radial_closure",
        Nmax=6,
        Mmax=6,
        Ncs=12,
        mainfield_kind="radial",
        ignore_PFAC=True,
        connect_hemispheres=False,
        dynamics_mode="full_induction",
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        least_squares_solver="svd",
        benchmark_mode=True,
        backend="numpy",
        dense_full_operators=True,
    )

    grid = dynamics.state.geometry.grid
    hall = np.zeros_like(grid.lat)
    pedersen = np.ones_like(grid.lat)
    dynamics.set_conductance(hall, pedersen, lat=grid.lat, lon=grid.lon, time=None)
    dynamics.set_jr(np.zeros_like(grid.lat), lat=grid.lat, lon=grid.lon, time=None)
    dynamics.state.update(dynamics.input_manager, np.float64(0), interpolation=True)
    return dynamics.state


def test_radial_isotropic_inertia_is_positive_scalar_identity() -> None:
    """For radial main field, inertia matrix should reduce to scalar identity."""
    state = _build_radial_isotropic_state()
    C = np.asarray(state.toroidal_matrices.inertia_matrix, dtype=float)

    diag = np.diag(C)
    offdiag = C - np.diag(diag)

    offdiag_rel = np.linalg.norm(offdiag) / max(np.linalg.norm(C), 1e-30)
    diag_rel_std = np.std(diag) / max(np.mean(np.abs(diag)), 1e-30)

    assert np.min(diag) > 0.0
    assert offdiag_rel < 1e-10
    assert diag_rel_std < 1e-10


def test_radial_isotropic_toroidal_closure_is_diffusive() -> None:
    """Stripped radial/isotropic toroidal self-feedback must be non-growing."""
    state = _build_radial_isotropic_state()
    N = state.solution_basis.index_length

    forcing = np.asarray(state.toroidal_matrices.E_to_dtjr_forcing_matrix, dtype=float)
    forcing_pol = forcing[:, :N]
    forcing_tor = forcing[:, N:]
    forcing_tor_rel = np.linalg.norm(forcing_tor) / max(np.linalg.norm(forcing_pol), 1e-30)
    assert forcing_tor_rel < 1e-12

    a00 = np.asarray(
        state.get_coupled_induction_blocks(source="dense", use_pinning=True)["dtpsi_from_psi"],
        dtype=float,
    )
    max_real_a00 = float(np.max(np.linalg.eigvals(a00).real))
    assert max_real_a00 <= 1e-12

    report = state.get_coupled_stability_report(source="dense", use_pinning=True)
    assert float(report["positive_real_count"]) == 0.0
    assert float(report["max_real"]) <= 1e-12
