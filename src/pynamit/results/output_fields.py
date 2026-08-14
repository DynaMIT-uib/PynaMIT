"""Evaluate simulation output fields on requested grids."""

import numpy as np
from kompe import SphericalTransform
from kompe.constants import MU0

from pynamit.results.field_maps import evaluate_JS_from_maps
from pynamit.results.grid_evaluation import transform_for_basis
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    joule_heating_from_current,
    pedersen_geometry_tensor,
)


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
    entry = simulation.data.output_series.get_entry(key, simulation.current_time)
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
    boundary_jr_to_JS = geometry.boundary_jr_to_gridded_JS_operator(horizontal_transform)
    induced_Br_to_JS = geometry.induced_Br_to_gridded_JS_operator(horizontal_transform)
    boundary_Br_operator = (
        geometry.boundary_Br_to_gridded_JS_operator(horizontal_transform)
        if boundary_Br is not None
        else None
    )
    return evaluate_JS_from_maps(
        boundary_jr,
        induced_Br,
        boundary_jr_to_JS=boundary_jr_to_JS,
        induced_Br_to_JS=induced_Br_to_JS,
        boundary_Br=boundary_Br,
        boundary_Br_to_JS=boundary_Br_operator,
    )


def evaluate_JS(simulation, transform, *, key=None):
    """Evaluate total horizontal JS."""
    entry = current_output_entry(simulation, key=key)
    boundary_Br = None
    if "boundary_Br" in simulation.inputs:
        boundary_entry = simulation.data.input_series.get_entry(
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


def evaluate_simulation_output(
    source, time, *, key=None, grid=None, transform=None, interpolation=False, include_derived=True
):
    """Evaluate saved physical output on one spherical grid.

    Parameters
    ----------
    source : Simulation or SimulationResults
        Live or persisted simulation containing output coefficients.
    time : float
        Simulation time in seconds after ``t0``.
    key : {'dynamic', 'equilibrium'}, optional
        Output stream. The dynamic stream is preferred when both exist.
    grid : SphericalGrid, optional
        Evaluation grid. Defaults to the simulation model grid.
    transform : SphericalTransform, optional
        Explicit transform whose grid is used for evaluation.
    interpolation : bool, optional
        Interpolate coefficients between stored output and input times.
    include_derived : bool, optional
        Include electric field, sheet current, equivalent-current
        stream function, and Joule heating when conductance is present.

    Returns
    -------
    dict
        SI-valued arrays in the evaluation grid's shape. ``theta``
        components point south and ``phi`` components point east.
    """
    from pynamit.results.input_projection import evaluate_projected_input
    from pynamit.results.simulation_results import SimulationResults
    from pynamit.simulation import Simulation

    if grid is not None and transform is not None:
        raise ValueError("Supply either grid or transform, not both.")

    if isinstance(source, Simulation):
        output_series = source.data.output_series
        input_series = source.data.input_series
        geometry = source.geometry
    elif isinstance(source, SimulationResults):
        output_series = source.load_output_series()
        input_series = None
        geometry = source.load_geometry()
    else:
        raise TypeError("source must be a Simulation or SimulationResults.")

    if key is None:
        key = "dynamic" if "dynamic" in output_series.datasets else "equilibrium"
    if key not in {"dynamic", "equilibrium"}:
        raise ValueError("key must be 'dynamic' or 'equilibrium'.")
    if key not in output_series.datasets:
        raise ValueError(f"No {key!r} output stream is available.")
    entry = output_series.get_entry(key, time, interpolation=interpolation)
    if entry is None:
        raise ValueError(f"No {key!r} output is available at t={float(time):.3f}.")

    if transform is None:
        target_grid = geometry.model_grid if grid is None else grid
        transform = SphericalTransform(geometry.horizontal_basis, target_grid)
    horizontal_transform = transform_for_basis(geometry.horizontal_basis, transform)

    values = {
        "induced_Br": evaluate_induced_Br_coefficients(geometry, entry["induced_Br"], transform),
        "boundary_jr": evaluate_boundary_jr_coefficients(
            geometry, entry["boundary_jr"], transform
        ),
        "Phi": evaluate_Phi_coefficients(geometry, entry["Phi"], transform),
        "W": evaluate_W_coefficients(geometry, entry["W"], transform),
    }
    if not include_derived:
        return values

    if input_series is None:
        input_series = source.load_input_series()

    E_theta, E_phi = horizontal_transform.synthesize_helmholtz(
        np.stack((entry["Phi"], entry["W"]))
    )
    values.update(
        {
            "E_theta": E_theta,
            "E_phi": E_phi,
            "E_mag": np.hypot(E_theta, E_phi),
            "equivalent_current_function": evaluate_equivalent_current_coefficients(
                geometry, entry["induced_Br"], transform
            ),
        }
    )

    boundary_Br = None
    if "boundary_Br" in input_series.datasets:
        boundary_entry = input_series.get_entry("boundary_Br", time, interpolation=interpolation)
        if boundary_entry is not None:
            boundary_Br = boundary_entry["boundary_Br"]
    sheet_current = evaluate_JS_coefficients(
        geometry, entry["boundary_jr"], entry["induced_Br"], transform, boundary_Br=boundary_Br
    )
    values.update(
        {
            "JS_theta": sheet_current[0],
            "JS_phi": sheet_current[1],
            "JS_mag": np.hypot(sheet_current[0], sheet_current[1]),
        }
    )

    if "conductance" in input_series.datasets:
        conductance = evaluate_projected_input(
            source, "conductance", time, transform=transform, interpolation=interpolation
        )
        from pynamit.geomagnetism import MagneticFieldEvaluation

        main_field = MagneticFieldEvaluation(geometry.main_field, transform.grid, geometry.RI)
        pedersen_geometry = pedersen_geometry_tensor(
            main_field.unit_btheta, main_field.unit_bphi, main_field.unit_br
        )
        values["joule_heating"] = joule_heating_from_current(
            sheet_current, conductance["etaP"], pedersen_geometry
        )
    return values


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
    "evaluate_simulation_output",
]
