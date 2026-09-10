"""Faraday-induction evolution of physical radial magnetic field."""

import numpy as np
from kompe.math import get_array_module, prepare_linear_evolution

DEFAULT_DT_SECONDS = 5e-4
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-12  # Tesla, applied to physical induced-Br coefficients.


def _poloidal_potential_time_derivative(response, potential, forcing_W):
    """Return the private potential-coordinate time derivative."""
    induced_W = response.induced_poloidal_potential_feedback_operator.matvec(potential)
    return response.geometry.induced_poloidal_potential_faraday_rate_scale * (
        induced_W + forcing_W
    )


def induced_Br_time_derivative(response, induced_Br, E_coeffs_noninductive):
    """Return d(induced_Br)/dt for the current response."""
    geometry = response.geometry
    potential = geometry.induced_Br_to_poloidal_potential_operator.matvec(induced_Br)
    surface_W = geometry.helmholtz_divergence_free_potential_operator.matvec(E_coeffs_noninductive)
    forcing_W = geometry.surface_to_poloidal_operator.matvec(surface_W)
    rate = _poloidal_potential_time_derivative(response, potential, forcing_W)
    return geometry.induced_poloidal_potential_to_Br_operator.matvec(rate)


def equilibrium_induced_Br(response, E_coeffs_noninductive):
    """Return equilibrium Br from a minimum-norm potential fit.

    The solve minimizes the poloidal W residual. Singular or truncated
    modes can leave a residual; inspect induced_Br_time_derivative for
    the estimate's remaining evolution. This diagnostic is not a
    prerequisite for any integrator.
    """
    W = response.geometry.helmholtz_divergence_free_potential_operator(E_coeffs_noninductive)
    return response.noninductive_W_to_equilibrium_induced_Br_operator(W)


def build_induction_stepper(
    response, E_coeffs_noninductive, *, dt=None, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL
):
    """Prepare Faraday evolution with fixed conductance and forcing.

    Return `advance(initial_Br, times, *, batch_size=32)`, an iterator
    of physical Br arrays with a trailing sample axis. `times` gives
    offsets from the initial state.
    The generic integration lives in Kompe; coordinate conversion and
    error tolerances belong here. `atol` is in tesla.
    Euler uses `dt` (default 0.5 ms); other methods need no fixed step.
    """
    integrator = response.config.integrator
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero.")

    geometry = response.geometry
    surface_W = geometry.helmholtz_divergence_free_potential_operator.matvec(E_coeffs_noninductive)
    forcing_W = geometry.surface_to_poloidal_operator.matvec(surface_W)
    rate_scale = float(geometry.induced_poloidal_potential_faraday_rate_scale)

    # Faraday's law in poloidal-potential coordinates: dP/dt = A P + b.
    A = rate_scale * response.induced_poloidal_potential_feedback_operator
    b = rate_scale * forcing_W
    xp = get_array_module(*A.backend_operands, b)
    potential_atol = atol
    if integrator not in ("euler", "exponential"):
        potential_atol = geometry.induced_Br_to_poloidal_potential_operator.matvec(
            xp.full(b.shape, atol)
        )
    advance_potential = prepare_linear_evolution(
        A,
        b,
        method=integrator,
        dt=DEFAULT_DT_SECONDS if integrator == "euler" and dt is None else dt,
        rtol=rtol,
        atol=potential_atol,
    )

    def advance(initial_Br, times, *, batch_size=32, output_interval=None):
        potential = geometry.induced_Br_to_poloidal_potential_operator.matvec(
            xp.asarray(initial_Br)
        )
        for samples in advance_potential(
            potential, times, batch_size=batch_size, output_interval=output_interval
        ):
            yield geometry.induced_poloidal_potential_to_Br_operator(samples)

    return advance


def evolve_induced_Br(
    response,
    induced_Br,
    duration,
    E_coeffs_noninductive,
    *,
    dt=None,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
):
    """Advance one physical field, retaining only the final state.

    For repeated or sampled evolution, reuse build_induction_stepper.
    duration is the physical interval; dt is Euler's internal step.
    """
    advance = build_induction_stepper(response, E_coeffs_noninductive, dt=dt, rtol=rtol, atol=atol)
    return next(advance(induced_Br, [duration]))[..., 0]


__all__ = [
    "build_induction_stepper",
    "equilibrium_induced_Br",
    "evolve_induced_Br",
    "induced_Br_time_derivative",
]
