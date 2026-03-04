"""Regression tests for stripped radial/isotropic toroidal closure behavior."""

from __future__ import annotations

import numpy as np

from pynamit.simulation.dynamics import Dynamics, SimulationMode


def _build_radial_isotropic_state():
    dynamics = Dynamics(
        run_directory="test_toroidal_radial_closure",
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


def test_radial_isotropic_inertia_vanishes_for_pure_radial_field() -> None:
    """For pure radial background field, |B_s|^2=0 so inertia must vanish."""
    state = _build_radial_isotropic_state()
    mass_dtalpha = np.asarray(state.toroidal_matrices.mass_dtalpha, dtype=float)
    assert np.linalg.norm(mass_dtalpha) < 1e-12


def test_radial_isotropic_toroidal_closure_is_diffusive() -> None:
    """Stripped radial/isotropic toroidal self-feedback must be non-growing."""
    state = _build_radial_isotropic_state()

    forcing = np.asarray(state.toroidal_matrices.toroidal_rhs_from_E_operator, dtype=float)
    # For a purely radial background field (B_theta = B_phi = 0), the
    # Er-free toroidal forcing projection vanishes identically.
    assert np.linalg.norm(forcing) < 1e-12

    a00 = np.asarray(
        state.get_coupled_induction_blocks(source="dense", use_pinning=True)["dt_psi_from_psi"],
        dtype=float,
    )
    max_real_a00 = float(np.max(np.linalg.eigvals(a00).real))
    assert max_real_a00 <= 1e-12

    report = state.get_coupled_stability_report(source="dense", use_pinning=True)
    assert float(report["positive_real_count"]) == 0.0
    assert float(report["max_real"]) <= 1e-12
