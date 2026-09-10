"""Evaluate simulation output fields on requested grids."""

from functools import cached_property

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

    ``source`` is live or saved SimulationResults. ``time``
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

    if not isinstance(source, SimulationResults):
        raise TypeError("source must be SimulationResults; use simulation.results for a live run.")
    series = source.load_output_series(key) if key is not None else source.load_output_series()
    key = select_output_stream(series.datasets, preferred=key)
    if time is None:
        coefficients = series.datasets[key][series.get_data_var_name(key, "induced_Br")].values.T
    else:
        times = np.atleast_1d(np.asarray(time, dtype=float))
        if times.ndim != 1 or times.size == 0:
            raise ValueError("time must be a scalar or non-empty 1-D array.")
        entry = series.get_entry(key, times, variables=("induced_Br",))
        if entry is None:
            raise ValueError("No output is available at one or more requested times.")
        coefficients = entry["induced_Br"]
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
    jr_map = as_linear_map(boundary_jr_to_JS)
    Br_map = as_linear_map(induced_Br_to_JS)
    batch_shape = np.shape(boundary_jr)[len(jr_map.input_shape) :]
    shape = (2, jr_map.shape[0] // 2) + batch_shape
    current = jr_map(boundary_jr).reshape(shape) + Br_map(induced_Br).reshape(shape)
    if boundary_Br is not None:
        if boundary_Br_to_JS is None:
            raise ValueError("boundary_Br_to_JS is required when boundary_Br is provided.")
        current = current + as_linear_map(boundary_Br_to_JS)(boundary_Br).reshape(shape)
    return current


class OutputEvaluation:
    """Reusable physical field maps on one simulation geometry/grid.

    ``evaluate`` accepts coefficients with trailing batch axes and
    returns SI-valued fields. Only requested maps are constructed.
    Geometry and materializations are reused, never sampled values.
    For histories, pass this object as ``evaluation`` to
    ``evaluate_simulation_output``.
    """

    def __init__(self, geometry, grid=None, *, transform=None):
        if grid is not None and transform is not None:
            raise ValueError("Supply either grid or transform, not both.")
        self.geometry = geometry
        self.transform = (
            transform.with_basis(geometry.horizontal_basis)
            if transform is not None
            else geometry.horizontal_transform
            if grid is None
            else SphericalTransform(geometry.horizontal_basis, grid)
        )
        self.grid = self.transform.grid

    @cached_property
    def poloidal_transform(self):
        """Scalar magnetic-field synthesis on the same grid."""
        return self.transform.with_basis(self.geometry.poloidal_basis)

    @cached_property
    def equivalent_current_operator(self):
        """Induced radial field to equivalent current function."""
        return (-self.geometry.RI / MU0) * (
            self.poloidal_transform.scalar_synthesis_operator
            @ self.geometry.poloidal_to_normalized_potential_jump_operator
            @ self.geometry.induced_Br_to_poloidal_potential_operator
        )

    @cached_property
    def sheet_current_operators(self):
        """Current contributions from induced and boundary fields."""
        geometry, transform = self.geometry, self.transform
        return {
            "induced_Br_to_JS": geometry.induced_Br_to_gridded_JS_operator(transform),
            "boundary_jr_to_JS": geometry.boundary_jr_to_gridded_JS_operator(transform),
            "boundary_Br_to_JS": geometry.boundary_Br_to_gridded_JS_operator(transform),
        }

    @cached_property
    def pedersen_geometry(self):
        """Dissipative geometry tensor in (theta, phi) components."""
        unit_br, unit_btheta, unit_bphi = self.geometry.main_field.unit_vector(
            self.grid, self.geometry.RI
        )
        return pedersen_geometry_tensor(unit_btheta, unit_bphi, unit_br)

    def evaluate(self, coefficients, *, field_names=None, boundary_Br=None, etaP=None):
        """Evaluate flat grid fields with trailing coefficient batches.

        Potentials Phi/W become volts; E is V/m, sheet current A/m,
        induced Br tesla, boundary jr A/m², equivalent current amperes,
        and Joule heating W/m². ``etaP`` supplies Pedersen resistance on
        this grid when Joule heating is requested.
        """
        requested = set(_OUTPUT_VALUE_NAMES) if field_names is None else set(field_names)
        if field_names is None and etaP is None:
            requested.remove("joule_heating")
        unknown = requested - _OUTPUT_VALUE_NAMES
        if unknown:
            raise ValueError(f"Unknown output fields requested: {sorted(unknown)}.")

        values = {}

        if "induced_Br" in requested:
            values["induced_Br"] = self.poloidal_transform.scalar_synthesis_operator(
                coefficients["induced_Br"]
            )
        if "boundary_jr" in requested:
            values["boundary_jr"] = self.transform.scalar_synthesis_operator(
                coefficients["boundary_jr"]
            )
        if "equivalent_current_function" in requested:
            values["equivalent_current_function"] = self.equivalent_current_operator(
                coefficients["induced_Br"]
            )

        potential_fields = requested & {"Phi", "W"}
        if potential_fields:
            radius = self.geometry.RI
            horizontal_transform = self.transform
        if "Phi" in potential_fields:
            values["Phi"] = radius * horizontal_transform.scalar_synthesis_operator(
                coefficients["Phi"]
            )
        if "W" in potential_fields:
            values["W"] = radius * horizontal_transform.scalar_synthesis_operator(
                coefficients["W"]
            )

        electric_fields = requested & {"E_theta", "E_phi", "E_mag"}
        if electric_fields:
            horizontal_transform = self.transform
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
            sheet_current = evaluate_sheet_current(
                coefficients["boundary_jr"],
                coefficients["induced_Br"],
                boundary_jr_to_JS=self.sheet_current_operators["boundary_jr_to_JS"],
                induced_Br_to_JS=self.sheet_current_operators["induced_Br_to_JS"],
                boundary_Br=boundary_Br,
                boundary_Br_to_JS=self.sheet_current_operators["boundary_Br_to_JS"],
            )
            xp = get_array_module(sheet_current)
            if "JS_theta" in requested:
                values["JS_theta"] = sheet_current[0]
            if "JS_phi" in requested:
                values["JS_phi"] = sheet_current[1]
            if "JS_mag" in requested:
                values["JS_mag"] = xp.hypot(sheet_current[0], sheet_current[1])
            if "joule_heating" in requested:
                if etaP is None:
                    raise ValueError("etaP is required to evaluate Joule heating.")
                # Geometry is fixed; time axes belong to J and etaP.
                tensor = xp.asarray(self.pedersen_geometry).reshape(
                    (2, 2, sheet_current.shape[1]) + (1,) * (sheet_current.ndim - 2)
                )
                values["joule_heating"] = joule_heating_from_current(sheet_current, etaP, tensor)
        return values


def evaluate_simulation_output(
    source,
    time,
    *,
    key=None,
    grid=None,
    transform=None,
    interpolation=False,
    field_names=None,
    evaluation=None,
):
    """Evaluate live or saved output as SI-valued fields on one grid.

    `source` is SimulationResults; `time` is seconds after its t0.
    `key` selects dynamic or equilibrium (dynamic is preferred).
    Supply a grid or a reusable transform; otherwise use the model grid.
    Scalar queries retain the grid's broadcast shape. A non-empty
    time array adds a trailing time axis, preserving query order.
    Default selection omits Joule heating if conductance is missing
    at any requested time; explicit requests then raise ValueError.

    `field_names` selects physical quantities, for example
    `{"induced_Br", "Phi", "joule_heating"}`. Omission evaluates all
    fields, excluding Joule heating when no conductance is available.
    Explicit Joule requests require conductance at the requested time.

    Supply ``evaluation=OutputEvaluation(geometry, grid)`` to reuse
    physical field maps across requests. It replaces grid/transform;
    maps are cached, never sampled field values.
    Input and output time selection shares FieldTimeSeries semantics:
    held values are causal, or linearly interpolated when requested.
    """
    from pynamit.results.input_fields import evaluate_projected_input
    from pynamit.results.simulation_results import SimulationResults

    if grid is not None and transform is not None:
        raise ValueError("Supply either grid or transform, not both.")
    if not isinstance(source, SimulationResults):
        raise TypeError("source must be SimulationResults; use simulation.results for a live run.")
    if evaluation is not None and (grid is not None or transform is not None):
        raise ValueError("Supply evaluation or grid/transform, not both.")
    if evaluation is None:
        evaluation = OutputEvaluation(source.geometry, grid, transform=transform)
    elif evaluation.geometry is not source.geometry:
        raise ValueError("evaluation must use this result's geometry.")
    transform = evaluation.transform
    requested = set(_OUTPUT_VALUE_NAMES if field_names is None else field_names)
    unknown = requested - _OUTPUT_VALUE_NAMES
    if unknown:
        raise ValueError(f"Unknown output fields requested: {sorted(unknown)}.")
    output_series = (
        source.load_output_series(key) if key is not None else source.load_output_series()
    )
    key = select_output_stream(output_series.datasets, preferred=key)
    required_coefficients = requested & {"induced_Br", "boundary_jr", "Phi", "W"}
    if requested & {
        "equivalent_current_function",
        "JS_theta",
        "JS_phi",
        "JS_mag",
        "joule_heating",
    }:
        required_coefficients |= {"induced_Br"}
    if requested & {"JS_theta", "JS_phi", "JS_mag", "joule_heating"}:
        required_coefficients |= {"boundary_jr"}
    if requested & {"E_theta", "E_phi", "E_mag"}:
        required_coefficients |= {"Phi", "W"}
    entry = output_series.get_entry(
        key, time, interpolation=interpolation, variables=required_coefficients
    )
    if entry is None:
        raise ValueError(f"No {key!r} output is available at one or more requested times.")

    boundary_Br = None
    if requested & {"JS_theta", "JS_phi", "JS_mag", "joule_heating"}:
        series = source.load_input_series("boundary_Br")
        boundary = (
            series.get_entry("boundary_Br", time, interpolation=interpolation, fill_value=0.0)
            if "boundary_Br" in series.datasets
            else None
        )
        if boundary is not None:
            boundary_Br = boundary["boundary_Br"]

    etaP = None
    if "joule_heating" in requested:
        series = source.load_input_series("conductance")
        conductance = series.get_entry("conductance", time, variables=())
        if conductance is None:
            if field_names is not None:
                raise ValueError(
                    "No conductance is available at one or more requested times for Joule heating."
                )
            requested.remove("joule_heating")
        else:
            etaP = evaluate_projected_input(
                series, "conductance", time, transform=transform, interpolation=interpolation
            )["etaP"]
    values = evaluation.evaluate(entry, field_names=requested, boundary_Br=boundary_Br, etaP=etaP)
    batch_shape = () if np.ndim(time) == 0 else (len(time),)
    return {
        name: array.reshape(transform.grid.shape + batch_shape) for name, array in values.items()
    }


__all__ = [
    "evaluate_sheet_current",
    "evaluate_ground_magnetic_field",
    "build_ground_magnetic_field_operators",
    "select_output_stream",
    "OutputEvaluation",
    "evaluate_simulation_output",
]
