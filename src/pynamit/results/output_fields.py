"""Evaluate simulation output fields on requested grids."""

from kompe import SphericalTransform
from kompe.constants import MU0
from kompe.math import get_array_module

from pynamit.results.evaluation import (
    apply_coefficient_operator,
    evaluate_sheet_current_from_operators,
    transform_for_basis,
)
from pynamit.simulation.electrodynamics.ionospheric_closure import (
    joule_heating_from_current,
    pedersen_geometry_tensor,
)

_OUTPUT_VALUE_NAMES = frozenset(
    {
        "induced_Br",
        "boundary_jr",
        "equivalent_current_function",
        "Phi",
        "W",
        "E_theta",
        "E_phi",
        "E_mag",
        "JS_theta",
        "JS_phi",
        "JS_mag",
        "joule_heating",
    }
)


def select_output_stream(datasets, preferred=None):
    """Select a dynamic or equilibrium output stream."""
    if preferred is not None:
        if preferred not in datasets:
            raise ValueError(f"No output dataset named {preferred!r} is available.")
        return preferred
    if "dynamic" in datasets:
        return "dynamic"
    if "equilibrium" in datasets:
        return "equilibrium"
    raise RuntimeError("No dynamic or equilibrium output is available to visualize.")


def output_at_current_time(simulation, key=None):
    """Return current output coefficients from a simulation."""
    key = select_output_stream(simulation.outputs, preferred=key)
    entry = simulation.data.output_series.get_entry(key, simulation.current_time)
    if entry is None:
        raise RuntimeError(
            f"No {key!r} output is available at t={float(simulation.current_time):.3f}."
        )
    return entry


def output_evaluation_operators(geometry, transform):
    """Return reusable coefficient-to-field operators for one grid."""
    horizontal_transform = transform_for_basis(geometry.horizontal_basis, transform)
    poloidal_transform = geometry.poloidal_transform_for(horizontal_transform)
    return {
        "RI": float(geometry.RI),
        "horizontal_transform": horizontal_transform,
        "induced_Br_to_Br": poloidal_transform.scalar_synthesis_operator,
        "boundary_jr_to_jr": horizontal_transform.scalar_synthesis_operator,
        "induced_Br_to_Jeq": (-float(geometry.RI) / MU0)
        * (
            poloidal_transform.scalar_synthesis_operator
            @ geometry.poloidal_to_boundary_potential_jump_factor_operator
            @ geometry.induced_Br_to_poloidal_potential_operator
        ),
    }


def sheet_current_operators(geometry, transform):
    """Return reusable sheet-current operators for one grid."""
    horizontal_transform = transform_for_basis(geometry.horizontal_basis, transform)
    poloidal_transform = geometry.poloidal_transform_for(horizontal_transform)
    return {
        "induced_Br_to_JS": geometry.induced_Br_to_gridded_JS_operator(
            horizontal_transform, poloidal_transform=poloidal_transform
        ),
        "boundary_jr_to_JS": geometry.boundary_jr_to_gridded_JS_operator(
            horizontal_transform, poloidal_transform=poloidal_transform
        ),
        "boundary_Br_to_JS": geometry.boundary_Br_to_gridded_JS_operator(
            horizontal_transform, poloidal_transform=poloidal_transform
        ),
    }


def evaluate_output_coefficients(
    coefficients,
    transform,
    *,
    geometry=None,
    field_names=None,
    operators=None,
    current_operators=None,
    boundary_Br=None,
    etaP=None,
    pedersen_geometry=None,
):
    """Evaluate one output coefficient row as SI-valued physical fields.

    Optional operator dictionaries let plotting and movie workflows
    reuse materialized maps across times without maintaining another
    implementation of the physical field evaluation.
    """
    requested = set(_OUTPUT_VALUE_NAMES) if field_names is None else set(field_names)
    if field_names is None and (etaP is None or pedersen_geometry is None):
        requested.remove("joule_heating")
    unknown = requested - _OUTPUT_VALUE_NAMES
    if unknown:
        raise ValueError(f"Unknown output fields requested: {sorted(unknown)}.")

    basic_fields = requested & {
        "induced_Br",
        "boundary_jr",
        "equivalent_current_function",
        "Phi",
        "W",
        "E_theta",
        "E_phi",
        "E_mag",
    }
    if operators is None and basic_fields:
        if geometry is None:
            raise ValueError("geometry is required when operators are not supplied.")
        operators = output_evaluation_operators(geometry, transform)
    values = {}

    if "induced_Br" in requested:
        values["induced_Br"] = apply_coefficient_operator(
            operators["induced_Br_to_Br"], coefficients["induced_Br"]
        )
    if "boundary_jr" in requested:
        values["boundary_jr"] = apply_coefficient_operator(
            operators["boundary_jr_to_jr"], coefficients["boundary_jr"]
        )
    if "equivalent_current_function" in requested:
        values["equivalent_current_function"] = apply_coefficient_operator(
            operators["induced_Br_to_Jeq"], coefficients["induced_Br"]
        )

    potential_fields = requested & {"Phi", "W"}
    if potential_fields:
        radius = float(operators["RI"])
        horizontal_transform = operators["horizontal_transform"]
    if "Phi" in potential_fields:
        values["Phi"] = radius * apply_coefficient_operator(
            horizontal_transform.scalar_synthesis_operator, coefficients["Phi"]
        )
    if "W" in potential_fields:
        values["W"] = radius * apply_coefficient_operator(
            horizontal_transform.scalar_synthesis_operator, coefficients["W"]
        )

    electric_fields = requested & {"E_theta", "E_phi", "E_mag"}
    if electric_fields:
        horizontal_transform = operators["horizontal_transform"]
        xp = get_array_module(coefficients["Phi"], coefficients["W"])
        E_theta, E_phi = horizontal_transform.synthesize_helmholtz(
            xp.stack((coefficients["Phi"], coefficients["W"]))
        )
        if "E_theta" in requested:
            values["E_theta"] = E_theta
        if "E_phi" in requested:
            values["E_phi"] = E_phi
        if "E_mag" in requested:
            values["E_mag"] = xp.hypot(E_theta, E_phi)

    current_fields = requested & {"JS_theta", "JS_phi", "JS_mag", "joule_heating"}
    if current_fields:
        if current_operators is None:
            if geometry is None:
                raise ValueError(
                    "geometry is required when sheet-current operators are not supplied."
                )
            current_operators = sheet_current_operators(geometry, transform)
        sheet_current = evaluate_sheet_current_from_operators(
            coefficients["boundary_jr"],
            coefficients["induced_Br"],
            boundary_jr_to_JS=current_operators["boundary_jr_to_JS"],
            induced_Br_to_JS=current_operators["induced_Br_to_JS"],
            boundary_Br=boundary_Br,
            boundary_Br_to_JS=current_operators["boundary_Br_to_JS"],
        )
        xp = get_array_module(sheet_current)
        if "JS_theta" in requested:
            values["JS_theta"] = sheet_current[0]
        if "JS_phi" in requested:
            values["JS_phi"] = sheet_current[1]
        if "JS_mag" in requested:
            values["JS_mag"] = xp.hypot(sheet_current[0], sheet_current[1])
        if "joule_heating" in requested:
            if etaP is None or pedersen_geometry is None:
                raise ValueError(
                    "etaP and pedersen_geometry are required to evaluate Joule heating."
                )
            values["joule_heating"] = joule_heating_from_current(
                sheet_current, etaP, pedersen_geometry
            )
    return values


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
    from pynamit.results.input_evaluation import evaluate_projected_input
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

    key = select_output_stream(output_series.datasets, preferred=key)
    entry = output_series.get_entry(key, time, interpolation=interpolation)
    if entry is None:
        raise ValueError(f"No {key!r} output is available at t={float(time):.3f}.")

    if transform is None:
        target_grid = geometry.model_grid if grid is None else grid
        transform = SphericalTransform(geometry.horizontal_basis, target_grid)
    basic_fields = {"induced_Br", "boundary_jr", "Phi", "W"}
    if not include_derived:
        return evaluate_output_coefficients(
            entry, transform, geometry=geometry, field_names=basic_fields
        )

    if input_series is None:
        input_series = source.load_input_series()

    boundary_Br = None
    if "boundary_Br" in input_series.datasets:
        boundary_entry = input_series.get_entry("boundary_Br", time, interpolation=interpolation)
        if boundary_entry is not None:
            boundary_Br = boundary_entry["boundary_Br"]
    field_names = basic_fields | {
        "E_theta",
        "E_phi",
        "E_mag",
        "equivalent_current_function",
        "JS_theta",
        "JS_phi",
        "JS_mag",
    }
    etaP = None
    geometry_tensor = None
    if "conductance" in input_series.datasets:
        conductance = evaluate_projected_input(
            source, "conductance", time, transform=transform, interpolation=interpolation
        )
        from pynamit.geomagnetism import MagneticFieldEvaluation

        main_field = MagneticFieldEvaluation(geometry.main_field, transform.grid, geometry.RI)
        etaP = conductance["etaP"]
        geometry_tensor = pedersen_geometry_tensor(
            main_field.unit_btheta, main_field.unit_bphi, main_field.unit_br
        )
        field_names.add("joule_heating")
    return evaluate_output_coefficients(
        entry,
        transform,
        geometry=geometry,
        field_names=field_names,
        boundary_Br=boundary_Br,
        etaP=etaP,
        pedersen_geometry=geometry_tensor,
    )


__all__ = [
    "output_at_current_time",
    "select_output_stream",
    "evaluate_output_coefficients",
    "evaluate_simulation_output",
    "output_evaluation_operators",
    "sheet_current_operators",
]
