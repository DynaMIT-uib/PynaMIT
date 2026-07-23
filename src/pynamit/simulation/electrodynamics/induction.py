"""Faraday-induction evolution for the poloidal magnetic state."""

from __future__ import annotations

import logging

from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pynamit.math.backend import to_jax, to_numpy, use_jax, xp

logger = logging.getLogger(__name__)


def m_ind_time_derivative(response, m_ind, E_coeffs_noninductive):
    """Return ``d(m_ind)/dt`` for the current ionospheric response."""
    poloidal_W = response.m_ind_feedback_operator.matvec(m_ind)
    surface_W_noninductive = response.geometry.helmholtz_divergence_free_potential_operator.matvec(
        E_coeffs_noninductive
    )
    poloidal_W += response.geometry.surface_to_poloidal_operator.matvec(surface_W_noninductive)
    return response.geometry.faraday_rate_scale * poloidal_W


def steady_state_m_ind(response, E_coeffs_noninductive):
    """Return the induced potential for zero time derivative."""
    E_noninductive_df = response.geometry.helmholtz_divergence_free_potential_operator.matvec(
        E_coeffs_noninductive
    )
    return response.noninductive_E_df_to_steady_m_ind_operator.matvec(E_noninductive_df)


def exponential_propagator(response, dt, *, m_ind_feedback_matrix=None):
    """Return the exact propagator for one constant-closure step."""
    if m_ind_feedback_matrix is None:
        m_ind_feedback_matrix = response.m_ind_feedback_matrix
    rate_matrix = response.geometry.faraday_rate_scale * m_ind_feedback_matrix
    return expm(float(dt) * to_numpy(rate_matrix))


def evolve_m_ind(
    response, m_ind, dt, E_coeffs_noninductive, steady_state=None, *, propagator=None
):
    """Advance induced-potential coefficients by one model time step."""
    backend_m_ind = xp.asarray(m_ind)
    backend_E_noninductive = xp.asarray(E_coeffs_noninductive)
    integrator = response.config.integrator

    if integrator == "euler":
        derivative = m_ind_time_derivative(response, backend_m_ind, backend_E_noninductive)
        return backend_m_ind + dt * derivative

    if integrator == "exponential":
        if steady_state is None:
            steady_state = steady_state_m_ind(response, backend_E_noninductive)
        if propagator is None:
            propagator = exponential_propagator(response, dt)
        difference = backend_m_ind - xp.asarray(steady_state)
        evolved = propagator @ to_numpy(difference) + to_numpy(steady_state)
        return to_jax(evolved) if use_jax() else evolved

    logger.debug("Using scipy.solve_ivp with method=%r.", integrator)
    m_ind_feedback = to_numpy(response.m_ind_feedback_matrix)
    E_noninductive_df = to_numpy(
        response.geometry.helmholtz_divergence_free_potential_operator.matvec(
            backend_E_noninductive
        )
    )
    poloidal_W_noninductive = to_numpy(
        response.geometry.surface_to_poloidal_operator.matvec(E_noninductive_df)
    )
    rate_scale = float(response.geometry.faraday_rate_scale)

    def rhs(_time, values):
        return rate_scale * (m_ind_feedback @ values + poloidal_W_noninductive)

    solution = solve_ivp(
        fun=rhs,
        t_span=(0, dt),
        y0=to_numpy(backend_m_ind),
        method=integrator,
        t_eval=[dt],
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(
            f"solve_ivp integrator {integrator!r} failed with status "
            f"{solution.status}: {solution.message}"
        )

    result = solution.y[:, -1]
    return to_jax(result) if use_jax() else result


__all__ = ["evolve_m_ind", "exponential_propagator", "m_ind_time_derivative", "steady_state_m_ind"]
