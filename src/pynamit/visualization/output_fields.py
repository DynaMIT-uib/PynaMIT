"""Evaluate simulation output fields on visualization grids."""

import numpy as np
from kompe.constants import MU0

from pynamit.visualization.field_maps import evaluate_JS_from_maps
from pynamit.visualization.grid_evaluation import transform_for_basis


def current_output_key(simulation, preferred=None):
    """Return the available output key to visualize."""
    datasets = simulation.outputs
    if preferred is not None:
        if preferred not in datasets:
            raise ValueError(f"No output dataset named {preferred!r} is available.")
        return preferred
    if "dynamic" in datasets:
        return "dynamic"
    if "equilibrium" in datasets:
        return "equilibrium"
    raise RuntimeError("No dynamic or equilibrium output is available to visualize.")


def current_output_entry(simulation, key=None):
    """Return current output coefficients from a simulation."""
    key = current_output_key(simulation, preferred=key)
    entry = simulation.run_data.output_series.get_entry(key, simulation.current_time)
    if entry is None:
        raise RuntimeError(
            f"No {key!r} output is available at t={float(simulation.current_time):.3f}."
        )
    return entry


def evaluate_induced_Br_coefficients(geometry, induced_Br, transform):
    """Evaluate the saved induced radial magnetic field."""
    return geometry.poloidal_transform_for(transform).synthesize_scalar(induced_Br)


def evaluate_induced_Br(simulation, transform, *, key=None):
    """Evaluate radial magnetic perturbation on ``transform.grid``."""
    entry = current_output_entry(simulation, key=key)
    return evaluate_induced_Br_coefficients(simulation.geometry, entry["induced_Br"], transform)


def evaluate_boundary_jr_coefficients(geometry, boundary_jr, transform):
    """Evaluate upper-boundary radial current density."""
    return transform_for_basis(geometry.horizontal_basis, transform).synthesize_scalar(boundary_jr)


def evaluate_boundary_jr(simulation, transform, *, key=None):
    """Evaluate radial current density on ``transform.grid``."""
    entry = current_output_entry(simulation, key=key)
    return evaluate_boundary_jr_coefficients(simulation.geometry, entry["boundary_jr"], transform)


def evaluate_equivalent_current_coefficients(geometry, induced_Br, transform):
    """Evaluate equivalent-current stream function from induced Br."""
    potential = geometry.induced_Br_to_poloidal_potential_operator.matvec(induced_Br)
    coeffs = (
        -geometry.RI
        / MU0
        * geometry.poloidal_to_boundary_potential_jump_factor_operator.matvec(potential)
    )
    return geometry.poloidal_transform_for(transform).synthesize_scalar(coeffs)


def evaluate_equivalent_current_function(simulation, transform, *, key=None):
    """Evaluate the equivalent-current stream function."""
    entry = current_output_entry(simulation, key=key)
    return evaluate_equivalent_current_coefficients(
        simulation.geometry, entry["induced_Br"], transform
    )


def evaluate_JS_coefficients(geometry, boundary_jr, induced_Br, transform, *, boundary_Br=None):
    """Evaluate total horizontal JS from coefficients."""
    boundary_jr = np.asarray(boundary_jr)
    induced_Br = np.asarray(induced_Br)

    horizontal_transform = transform_for_basis(geometry.horizontal_basis, transform)
    boundary_jr_to_JS = geometry.boundary_jr_to_gridded_JS(horizontal_transform)
    induced_Br_to_JS = geometry.induced_Br_to_gridded_JS(horizontal_transform)
    boundary_Br_to_JS = (
        geometry.boundary_Br_to_gridded_JS(horizontal_transform)
        if boundary_Br is not None
        else None
    )
    return evaluate_JS_from_maps(
        boundary_jr,
        induced_Br,
        boundary_jr_to_JS=boundary_jr_to_JS,
        induced_Br_to_JS=induced_Br_to_JS,
        boundary_Br=boundary_Br,
        boundary_Br_to_JS=boundary_Br_to_JS,
    )


def evaluate_JS(simulation, transform, *, key=None):
    """Evaluate total horizontal JS."""
    entry = current_output_entry(simulation, key=key)
    boundary_Br = None
    if "boundary_Br" in simulation.inputs:
        boundary_entry = simulation.run_data.input_series.get_entry(
            "boundary_Br", simulation.current_time
        )
        if boundary_entry is not None:
            boundary_Br = boundary_entry["boundary_Br"]
    elif simulation.response.boundary_Br is not None:
        boundary_Br = simulation.response.boundary_Br
    return evaluate_JS_coefficients(
        simulation.geometry,
        entry["boundary_jr"],
        entry["induced_Br"],
        transform,
        boundary_Br=boundary_Br,
    )


def evaluate_Phi_coefficients(geometry, Phi, transform):
    """Evaluate saved curl-free E coefficients as potential in volts."""
    return geometry.RI * transform_for_basis(
        geometry.horizontal_basis, transform
    ).synthesize_scalar(Phi)


def evaluate_Phi(simulation, transform, *, key=None):
    """Evaluate electric curl-free potential in volts."""
    entry = current_output_entry(simulation, key=key)
    return evaluate_Phi_coefficients(simulation.geometry, entry["Phi"], transform)


def evaluate_W_coefficients(geometry, W, transform):
    """Evaluate divergence-free E coefficients as potential in volts."""
    return geometry.RI * transform_for_basis(
        geometry.horizontal_basis, transform
    ).synthesize_scalar(W)


def evaluate_W(simulation, transform, *, key=None):
    """Evaluate electric divergence-free potential in volts."""
    entry = current_output_entry(simulation, key=key)
    return evaluate_W_coefficients(simulation.geometry, entry["W"], transform)


__all__ = [
    "current_output_entry",
    "current_output_key",
    "evaluate_JS",
    "evaluate_JS_coefficients",
    "evaluate_Phi",
    "evaluate_Phi_coefficients",
    "evaluate_W",
    "evaluate_W_coefficients",
    "evaluate_boundary_jr",
    "evaluate_boundary_jr_coefficients",
    "evaluate_equivalent_current_coefficients",
    "evaluate_equivalent_current_function",
    "evaluate_induced_Br",
    "evaluate_induced_Br_coefficients",
]
