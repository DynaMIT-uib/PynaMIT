"""Faraday-induction evolution of physical radial magnetic field."""

from __future__ import annotations

import logging

from kompe.math import to_jax, to_numpy, use_jax, xp
from scipy.integrate import solve_ivp
from scipy.linalg import expm

logger = logging.getLogger(__name__)


def _poloidal_potential_time_derivative(
    response, induced_poloidal_potential, E_coeffs_noninductive
):
    """Return the private potential-coordinate time derivative."""
    poloidal_W = response.induced_poloidal_potential_feedback_operator.matvec(
        induced_poloidal_potential
    )
    surface_W_noninductive = response.geometry.helmholtz_divergence_free_potential_operator.matvec(
        E_coeffs_noninductive
    )
    poloidal_W += response.geometry.surface_to_poloidal_operator.matvec(surface_W_noninductive)
    return response.geometry.induced_poloidal_potential_faraday_rate_scale * poloidal_W


def induced_Br_time_derivative(response, induced_Br, E_coeffs_noninductive):
    """Return ``d(induced_Br)/dt`` for the current response."""
    potential = response.geometry.induced_Br_to_poloidal_potential_operator.matvec(induced_Br)
    potential_rate = _poloidal_potential_time_derivative(
        response, potential, E_coeffs_noninductive
    )
    return response.geometry.induced_poloidal_potential_to_Br_operator.matvec(potential_rate)


def equilibrium_induced_Br(response, E_coeffs_noninductive):
    """Return induced Br for zero Faraday time derivative."""
    E_noninductive_df = response.geometry.helmholtz_divergence_free_potential_operator.matvec(
        E_coeffs_noninductive
    )
    return response.noninductive_E_df_to_equilibrium_induced_Br_operator.matvec(E_noninductive_df)


def poloidal_potential_exponential_propagator(response, dt, *, feedback_matrix=None):
    """Return the exact propagator in private potential coordinates."""
    if feedback_matrix is None:
        feedback_matrix = response.induced_poloidal_potential_feedback_matrix
    rate_matrix = response.geometry.induced_poloidal_potential_faraday_rate_scale * feedback_matrix
    return expm(float(dt) * to_numpy(rate_matrix))


def evolve_induced_Br(
    response,
    induced_Br,
    dt,
    E_coeffs_noninductive,
    equilibrium=None,
    *,
    poloidal_potential_propagator=None,
):
    """Advance physical induced-Br coefficients by one model time step.

    Time integration is performed in the better-conditioned private
    poloidal-potential coordinate and converted exactly at the API
    boundary.
    """
    geometry = response.geometry
    backend_induced_Br = xp.asarray(induced_Br)
    potential = geometry.induced_Br_to_poloidal_potential_operator.matvec(backend_induced_Br)
    backend_E_noninductive = xp.asarray(E_coeffs_noninductive)
    integrator = response.config.integrator

    if integrator == "euler":
        potential_rate = _poloidal_potential_time_derivative(
            response, potential, backend_E_noninductive
        )
        evolved_potential = potential + dt * potential_rate
        return geometry.induced_poloidal_potential_to_Br_operator.matvec(evolved_potential)

    if integrator == "exponential":
        if equilibrium is None:
            equilibrium = equilibrium_induced_Br(response, backend_E_noninductive)
        equilibrium_potential = geometry.induced_Br_to_poloidal_potential_operator.matvec(
            xp.asarray(equilibrium)
        )
        if poloidal_potential_propagator is None:
            poloidal_potential_propagator = poloidal_potential_exponential_propagator(response, dt)
        difference = potential - equilibrium_potential
        evolved_potential = poloidal_potential_propagator @ to_numpy(difference) + to_numpy(
            equilibrium_potential
        )
        evolved_Br = geometry.induced_poloidal_potential_to_Br_operator.matvec(evolved_potential)
        return to_jax(evolved_Br) if use_jax() else evolved_Br

    logger.debug("Using scipy.solve_ivp with method=%r.", integrator)
    feedback = to_numpy(response.induced_poloidal_potential_feedback_matrix)
    E_noninductive_df = to_numpy(
        geometry.helmholtz_divergence_free_potential_operator.matvec(backend_E_noninductive)
    )
    poloidal_W_noninductive = to_numpy(
        geometry.surface_to_poloidal_operator.matvec(E_noninductive_df)
    )
    rate_scale = float(geometry.induced_poloidal_potential_faraday_rate_scale)

    def rhs(_time, values):
        return rate_scale * (feedback @ values + poloidal_W_noninductive)

    solution = solve_ivp(
        fun=rhs,
        t_span=(0, dt),
        y0=to_numpy(potential),
        method=integrator,
        t_eval=[dt],
        dense_output=False,
    )
    if not solution.success:
        raise RuntimeError(
            f"solve_ivp integrator {integrator!r} failed with status "
            f"{solution.status}: {solution.message}"
        )

    evolved_Br = geometry.induced_poloidal_potential_to_Br_operator.matvec(solution.y[:, -1])
    return to_jax(evolved_Br) if use_jax() else evolved_Br


__all__ = [
    "equilibrium_induced_Br",
    "evolve_induced_Br",
    "induced_Br_time_derivative",
    "poloidal_potential_exponential_propagator",
]
