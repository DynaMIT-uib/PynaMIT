"""Evaluate simulation output fields on requested grids."""

import numpy as np
from kompe import SphericalTransform
from kompe.constants import EARTH_RADIUS_M, MU0
from kompe.math import (
    as_linear_map,
    diagonal_linear_map,
    get_array_module,
    pointwise_component_map,
)

from pynamit.results.field_evaluation import (
    model_grid_from_geographic,
    model_to_geographic_tangential_array,
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


def build_output_evaluation_operators(geometry, transform, *, include_joule=False):
    """Build field maps and optional Joule geometry for one grid."""
    horizontal_transform = transform.with_basis(geometry.horizontal_basis)
    poloidal_transform = transform.with_basis(geometry.poloidal_basis)
    operators = {
        "RI": float(geometry.RI),
        "horizontal_transform": horizontal_transform,
        "induced_Br_to_Br": poloidal_transform.scalar_synthesis_operator,
        "boundary_jr_to_jr": horizontal_transform.scalar_synthesis_operator,
        "induced_Br_to_Jeq": (-float(geometry.RI) / MU0)
        * (
            poloidal_transform.scalar_synthesis_operator
            @ geometry.poloidal_to_normalized_potential_jump_operator
            @ geometry.induced_Br_to_poloidal_potential_operator
        ),
    }
    if include_joule:
        unit_br, unit_btheta, unit_bphi = geometry.main_field.unit_vector(
            transform.grid, geometry.RI
        )
        operators["pedersen_geometry"] = pedersen_geometry_tensor(unit_btheta, unit_bphi, unit_br)
    return operators


def build_ground_magnetic_field_operators(
    geometry, grid, *, ground_radius=EARTH_RADIUS_M, coordinate_system="model"
):
    """Build induced-Br-to-ground-magnetic-field operators.

    The ionospheric induced radial field is continued inward as a
    regular solid harmonic from ``geometry.RI`` to ``ground_radius``.
    The returned radial and tangential operators evaluate components
    in tesla, with tangential component order ``(theta, phi)``
    (south, east).

    ``coordinate_system`` names both the input grid's frame and the
    returned components' frame: ``model`` or geocentric ``geographic``.
    """
    if coordinate_system not in {"model", "geographic"}:
        raise ValueError("coordinate_system must be 'model' or 'geographic'.")
    model_grid = (
        model_grid_from_geographic(geometry.main_field, grid.lat, grid.lon)
        if coordinate_system == "geographic"
        else grid
    )
    ionosphere_radius = float(geometry.RI)
    solid_harmonics = geometry.solid_harmonics
    basis = solid_harmonics.basis
    transform = SphericalTransform(basis, model_grid)

    reference_shift = solid_harmonics.regular_reference_shift_factors(
        ionosphere_radius, float(ground_radius)
    )
    radial_coefficient_shift = diagonal_linear_map(reference_shift)
    tangential_coefficient_shift = diagonal_linear_map(reference_shift / basis.n)

    tangential = transform.surface_gradient_operator @ tangential_coefficient_shift
    if coordinate_system == "geographic":
        rotation = model_to_geographic_tangential_array(geometry.main_field, model_grid)
        tangential = pointwise_component_map(rotation) @ tangential
    return {
        "radial": transform.scalar_synthesis_operator @ radial_coefficient_shift,
        "tangential": tangential,
    }


def evaluate_ground_magnetic_field(
    source,
    time=None,
    *,
    grid,
    key="dynamic",
    coordinate_system="geographic",
    ground_radius=EARTH_RADIUS_M,
    operators=None,
):
    """Evaluate the induced ground-field contribution in tesla.

    ``source`` is a live Simulation or saved SimulationResults. ``time``
    selects model seconds (scalar or 1-D); None selects all saved times
    in ``key``. Input coefficients are selected without interpolation.
    Arrays retain ``grid.shape`` followed by a time axis; tangential
    arrays additionally lead with ``(theta, phi)`` components.

    The grid and returned components use ``coordinate_system``. This
    continues induced Br inward; it does not add the background field.
    Repeated calls may reuse operators built for the same geometry,
    grid, frame and radius with build_ground_magnetic_field_operators.
    """
    from pynamit.results.simulation_results import SimulationResults
    from pynamit.simulation import Simulation

    if isinstance(source, SimulationResults):
        series = source.load_output_series(key) if key is not None else source.load_output_series()
    elif isinstance(source, Simulation):
        series = source.data.output_series
    else:
        raise TypeError("source must be a Simulation or SimulationResults.")
    key = select_output_stream(series.datasets, preferred=key)
    if time is None:
        coefficients = series.datasets[key][series.get_data_var_name(key, "induced_Br")].values.T
    else:
        times = np.atleast_1d(np.asarray(time, dtype=float))
        if times.ndim != 1 or times.size == 0:
            raise ValueError("time must be a scalar or non-empty 1-D array.")
        entries = [series.get_entry(key, value) for value in times]
        if any(entry is None for entry in entries):
            raise ValueError("No output is available at one or more requested times.")
        coefficients = np.stack([entry["induced_Br"] for entry in entries], axis=-1)
    if operators is None:
        operators = build_ground_magnetic_field_operators(
            source.geometry, grid, ground_radius=ground_radius, coordinate_system=coordinate_system
        )
    n_times = coefficients.shape[-1]
    return {
        "radial": operators["radial"].matmat(coefficients).reshape(grid.shape + (n_times,)),
        "tangential": operators["tangential"]
        .matmat(coefficients)
        .reshape((2,) + grid.shape + (n_times,)),
    }


def build_sheet_current_operators(geometry, transform):
    """Build reusable sheet-current operators for one grid."""
    horizontal_transform = transform.with_basis(geometry.horizontal_basis)
    poloidal_transform = transform.with_basis(geometry.poloidal_basis)
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


def evaluate_sheet_current(
    boundary_jr,
    induced_Br,
    *,
    boundary_jr_to_JS,
    induced_Br_to_JS,
    boundary_Br=None,
    boundary_Br_to_JS=None,
):
    """Evaluate sheet current from physical magnetic quantities."""
    current = as_linear_map(boundary_jr_to_JS).matvec(boundary_jr) + as_linear_map(
        induced_Br_to_JS
    ).matvec(induced_Br)
    if boundary_Br is not None:
        if boundary_Br_to_JS is None:
            raise ValueError("boundary_Br_to_JS is required when boundary_Br is provided.")
        current += as_linear_map(boundary_Br_to_JS).matvec(boundary_Br)
    return current.reshape(2, -1)


def evaluate_output_coefficients(
    coefficients,
    transform,
    *,
    geometry=None,
    field_names=None,
    operators=None,
    sheet_current_operators=None,
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
        operators = build_output_evaluation_operators(geometry, transform)
    values = {}

    if "induced_Br" in requested:
        values["induced_Br"] = as_linear_map(operators["induced_Br_to_Br"])(
            coefficients["induced_Br"]
        )
    if "boundary_jr" in requested:
        values["boundary_jr"] = as_linear_map(operators["boundary_jr_to_jr"])(
            coefficients["boundary_jr"]
        )
    if "equivalent_current_function" in requested:
        values["equivalent_current_function"] = as_linear_map(operators["induced_Br_to_Jeq"])(
            coefficients["induced_Br"]
        )

    potential_fields = requested & {"Phi", "W"}
    if potential_fields:
        radius = float(operators["RI"])
        horizontal_transform = operators["horizontal_transform"]
    if "Phi" in potential_fields:
        values["Phi"] = radius * horizontal_transform.scalar_synthesis_operator(
            coefficients["Phi"]
        )
    if "W" in potential_fields:
        values["W"] = radius * horizontal_transform.scalar_synthesis_operator(coefficients["W"])

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
        if sheet_current_operators is None:
            if geometry is None:
                raise ValueError(
                    "geometry is required when sheet-current operators are not supplied."
                )
            sheet_current_operators = build_sheet_current_operators(geometry, transform)
        sheet_current = evaluate_sheet_current(
            coefficients["boundary_jr"],
            coefficients["induced_Br"],
            boundary_jr_to_JS=sheet_current_operators["boundary_jr_to_JS"],
            induced_Br_to_JS=sheet_current_operators["induced_Br_to_JS"],
            boundary_Br=boundary_Br,
            boundary_Br_to_JS=sheet_current_operators["boundary_Br_to_JS"],
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
    from pynamit.results.input_fields import evaluate_projected_input
    from pynamit.results.simulation_results import SimulationResults
    from pynamit.simulation import Simulation

    if grid is not None and transform is not None:
        raise ValueError("Supply either grid or transform, not both.")

    if isinstance(source, Simulation):
        output_series = source.data.output_series
        input_series = source.data.input_series
        geometry = source.geometry
    elif isinstance(source, SimulationResults):
        output_series = (
            source.load_output_series(key) if key is not None else source.load_output_series()
        )
        input_series = None
        geometry = source.geometry
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
        input_series = source.load_input_series("boundary_Br", "conductance")

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
    if "conductance" in input_series.datasets:
        conductance = evaluate_projected_input(
            source, "conductance", time, transform=transform, interpolation=interpolation
        )
        etaP = conductance["etaP"]
        field_names.add("joule_heating")
    operators = build_output_evaluation_operators(
        geometry, transform, include_joule="joule_heating" in field_names
    )
    return evaluate_output_coefficients(
        entry,
        transform,
        geometry=geometry,
        field_names=field_names,
        operators=operators,
        boundary_Br=boundary_Br,
        etaP=etaP,
        pedersen_geometry=operators.get("pedersen_geometry"),
    )


__all__ = [
    "evaluate_sheet_current",
    "evaluate_ground_magnetic_field",
    "build_ground_magnetic_field_operators",
    "build_output_evaluation_operators",
    "build_sheet_current_operators",
    "output_at_current_time",
    "select_output_stream",
    "evaluate_output_coefficients",
    "evaluate_simulation_output",
]
