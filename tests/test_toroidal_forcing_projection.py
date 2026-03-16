"""Regression tests for toroidal forcing projection consistency."""

from __future__ import annotations

import numpy as np

from pynamit.simulation.settings import SimulationMode
from pynamit.simulation.settings import DynamicsMode, MainfieldKind


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values).reshape(-1)
    w = np.asarray(weights).reshape(-1)
    return float(np.sqrt(np.sum(w * vals * vals) / np.sum(w)))


def _forcing_rms_for_smooth_psi(
    sim_mode: SimulationMode, *, ncs: int = 10, northern_apex: bool = False
) -> float:
    from pynamit.simulation.runner import run_pynamit
    from pynamit.simulation.spatial import to_dense

    sim = run_pynamit(
        final_time=2.0,
        dt=1.0,
        plotsteps=1,
        Nmax=10,
        Mmax=5,
        Ncs=ncs,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=sim_mode.value,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
        northern_hemisphere_apex_constraints=northern_apex,
    )
    state = sim.state
    basis = state.solution_space
    grid = state.geometry.grid
    n = basis.index_length

    weights = np.asarray(grid.weights).reshape(-1)
    theta = np.deg2rad(np.asarray(grid.theta).reshape(-1))
    psi_grid = np.cos(theta)
    psi_coeffs = (
        np.asarray(basis.from_grid_values(psi_grid, grid, vector_type="scalar")).reshape(-1).copy()
    )

    psi_eval = np.asarray(basis.evaluate(psi_coeffs, grid, vector_type="scalar")).reshape(-1)
    psi_coeffs *= 1.0 / max(_weighted_rms(psi_eval, weights), 1e-30)

    psi_to_E = np.asarray(to_dense(state.toroidal_to_E_coeffs)).reshape(2 * n, n)
    E_coeffs = (psi_to_E @ psi_coeffs).reshape(2, n)
    forcing_coeffs = np.asarray(
        state.toroidal_matrices.toroidal_rhs_from_E_operator
    ) @ E_coeffs.reshape(-1)
    forcing_grid = np.asarray(basis.evaluate(forcing_coeffs, grid, vector_type="scalar")).reshape(
        -1
    )
    return _weighted_rms(forcing_grid, weights)


def test_cs_dominant_toroidal_forcing_not_area_suppressed() -> None:
    """CS-dominant forcing amplitude should be comparable to transform-CS.

    This guards against accidental area-weight scaling in coefficient-space
    forcing projection (which can suppress forcing by ~cell area).
    """
    forcing_st = _forcing_rms_for_smooth_psi(
        SimulationMode.SPECTRAL_TRANSFORM_CS, northern_apex=False
    )
    forcing_cs = _forcing_rms_for_smooth_psi(SimulationMode.CS_DOMINANT, northern_apex=True)
    ratio = forcing_cs / max(forcing_st, 1e-30)

    # Broad tolerance: implementations differ (global SH-like vs grid-dominant),
    # but they should remain the same order of magnitude.
    assert 0.1 <= ratio <= 10.0


def test_cs_dominant_uses_auxiliary_sh_toroidal_closure_basis() -> None:
    """Full-induction cs_dominant should use SH auxiliary closure basis."""
    from pynamit.simulation.runner import run_pynamit

    sim = run_pynamit(
        final_time=1.0,
        dt=1.0,
        plotsteps=1,
        Nmax=8,
        Mmax=4,
        Ncs=10,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.CS_DOMINANT,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        multi_data=True,
        connect_hemispheres=True,
        least_squares_solver="svd",
        northern_hemisphere_apex_constraints=True,
        use_jr=False,
        wind=False,
    )
    tor = sim.state.toroidal_matrices
    assert tor is not None
    assert getattr(tor.closure_derivative_basis, "kind", "") == "SH"
    assert getattr(tor.rhs_derivative_basis, "kind", "") == "SH"
    assert getattr(tor.radial_derivative_basis, "kind", "") == "SH"
