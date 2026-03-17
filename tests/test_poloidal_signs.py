"""Focused sign checks for induced poloidal operators."""

from __future__ import annotations

import numpy as np

from pynamit.simulation.runner import run_pynamit
from pynamit.simulation.settings import DynamicsMode, IntegratorKind, MainfieldKind, SimulationMode


def _build_state():
    sim = run_pynamit(
        final_time=0.0,
        dt=1.0,
        Nmax=8,
        Mmax=4,
        Ncs=10,
        dynamics_mode=DynamicsMode.FULL_INDUCTION,
        simulation_mode=SimulationMode.PURE_SPECTRAL,
        ignore_PFAC=False,
        mainfield_kind=MainfieldKind.IGRF,
        mainfield_epoch=2020,
        use_jr=False,
        wind=False,
        connect_hemispheres=False,
        benchmark_mode=True,
        dense_full_operators=False,
        integrator=IntegratorKind.EULER,
        least_squares_solver="svd",
    )
    return sim.state


def test_e_df_to_dm_ind_dt_matches_closed_form_sh_sign() -> None:
    """Faraday should give ``dm_ind/dt = -(1/RI) * E_df`` for the repo df basis."""
    state = _build_state()
    basis = state.solution_space

    degrees = np.asarray(basis.n, dtype=float).reshape(-1)
    nonzero = degrees > 0.0

    # Repo basis:
    #   Br_nm = n(n+1) m_ind_nm
    #   (curl E)_r,nm = n(n+1) E_df_nm / RI
    # and Faraday gives dBr/dt = -(curl E)_r.
    expected = -np.ones(np.count_nonzero(nonzero), dtype=float) / float(state.RI)
    derived = (
        -(degrees[nonzero] * (degrees[nonzero] + 1.0) / float(state.RI))
        / (degrees[nonzero] * (degrees[nonzero] + 1.0))
    )

    np.testing.assert_allclose(derived, expected, rtol=0.0, atol=1e-18)
    assert state.poloidal_matrices.E_df_to_d_m_ind_dt == expected[0]
