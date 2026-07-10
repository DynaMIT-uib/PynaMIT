"""Faraday-induction evolution for the magnetic state."""

from __future__ import annotations

import logging

from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pynamit.math.backend import to_jax, to_numpy, use_jax, xp

logger = logging.getLogger(__name__)


def m_ind_time_derivative(state, m_ind, E_coeffs_noind):
    """Return ``d(m_ind)/dt`` for the current ionospheric response."""
    E_df_total = state.m_ind_to_E_df_operator.matvec(m_ind)
    E_df_total += state.geometry.helmholtz_divergence_free_potential_operator.matvec(
        E_coeffs_noind
    )
    return state.geometry.E_df_to_d_m_ind_dt * E_df_total


def steady_state_m_ind(state, E_coeffs_noind):
    """Return the induced potential for zero time derivative."""
    E_noind_df = state.geometry.helmholtz_divergence_free_potential_operator.matvec(
        E_coeffs_noind
    )
    steady = state.E_noind_to_m_ind_steady_operator.matvec(E_noind_df)
    return state.project_scalar_mean_free(steady)


def evolve_m_ind(state, m_ind, dt, E_coeffs_noind, steady_state=None):
    """Advance induced-potential coefficients by one model time step."""
    backend_m_ind = xp.asarray(m_ind)
    backend_E_noind = xp.asarray(E_coeffs_noind)

    if state.integrator == "euler":
        derivative = m_ind_time_derivative(state, backend_m_ind, backend_E_noind)
        return state.project_scalar_mean_free(backend_m_ind + dt * derivative)

    if state.integrator == "exponential":
        operator = xp.asarray(
            state.geometry.E_df_to_d_m_ind_dt * state.m_ind_to_E_df_matrix
        )
        if steady_state is None:
            steady_state = steady_state_m_ind(state, backend_E_noind)
        difference = backend_m_ind - xp.asarray(steady_state)
        evolved = (
            expm(dt * to_numpy(operator)) @ to_numpy(difference)
            + to_numpy(steady_state)
        )
        return state.project_scalar_mean_free(evolved)

    logger.debug("Using scipy.solve_ivp with method=%r.", state.integrator)
    m_ind_to_E_df = to_numpy(state.m_ind_to_E_df_matrix)
    E_noind_df = to_numpy(
        state.geometry.helmholtz_divergence_free_potential_operator.matvec(backend_E_noind)
    )
    rate_scale = float(state.geometry.E_df_to_d_m_ind_dt)

    def rhs(_time, values):
        return rate_scale * (m_ind_to_E_df @ values + E_noind_df)

    solution = solve_ivp(
        fun=rhs,
        t_span=(0, dt),
        y0=to_numpy(backend_m_ind),
        method=state.integrator,
        t_eval=[dt],
        dense_output=False,
    )
    if not solution.success:
        logger.warning(
            "solve_ivp integrator %r failed with status %s: %s",
            state.integrator,
            solution.status,
            solution.message,
        )

    result = state.project_scalar_mean_free(solution.y[:, -1])
    return to_jax(result) if use_jax() else result


__all__ = ["evolve_m_ind", "m_ind_time_derivative", "steady_state_m_ind"]
