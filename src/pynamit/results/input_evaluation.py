"""Evaluate projected simulation inputs on spherical grids."""

from kompe.spherical_transform import SphericalTransform

from pynamit.fields import FieldCoefficients
from pynamit.results.evaluation import (
    evaluate_conductance_values,
    evaluate_tangential_coefficients,
)
from pynamit.storage import FieldTimeSeries


def evaluate_projected_input(
    source, key, time, *, grid=None, transform=None, interpolation=False, include_derived=True
):
    """Evaluate one projected input time series on a grid.

    Parameters
    ----------
    source : InputPreparation, Simulation, SimulationResults, or
        FieldTimeSeries
        Object containing projected input coefficient time series.
    key : str
        Input key, for example ``"boundary_jr"``, ``"boundary_Br"``,
        ``"conductance"``, ``"u"``, ``"Q_eff"``, or
        ``"E_neutral_wind"``.
    time : float
        Time value to select from the input time series.
    grid : SphericalGrid, optional
        Target grid. Required unless ``transform`` is supplied or
        ``source`` is an ``InputPreparation`` or ``Simulation`` with a
        model grid.
    transform : SphericalTransform, optional
        Explicit transform to use for evaluation.
    interpolation : bool, optional
        Whether to interpolate between stored input times.
    include_derived : bool, optional
        Include derived physical quantities such as conductances and
        vector magnitudes.

    Returns
    -------
    dict
        Evaluated input values keyed by variable/component name.
    """
    if grid is not None and transform is not None:
        raise ValueError("Supply either grid or transform, not both.")

    if isinstance(source, FieldTimeSeries):
        series = source
        default_grid = None
    else:
        from pynamit.results.simulation_results import SimulationResults
        from pynamit.simulation import InputPreparation

        if isinstance(source, InputPreparation):
            series = source.data.input_series
            default_grid = source.model_grid
        elif isinstance(source, SimulationResults):
            series = source.load_input_series()
            default_grid = source.schema.cs_basis.mesh.cell_centers
        else:
            raise TypeError(
                "source must be an InputPreparation, Simulation, "
                "SimulationResults, or FieldTimeSeries."
            )
    entry = series.get_entry(key, time, interpolation=interpolation)
    if entry is None:
        raise ValueError(f"No {key!r} input is available at t={float(time):.3f}.")

    field_space = series.get_field_space(key)
    target_grid = default_grid if grid is None else grid
    if transform is not None:
        transform = transform.with_basis(field_space.basis)
    elif target_grid is not None:
        transform = SphericalTransform(field_space.basis, target_grid)
    else:
        raise ValueError("A target grid or transform is required.")

    values = {}
    if field_space.field_type == "tangential":
        for var, coeffs in entry.items():
            field = FieldCoefficients(field_space, coeffs=coeffs)
            components = evaluate_tangential_coefficients(
                transform, field, include_magnitude=include_derived
            )
            values[f"{var}_theta"] = components["theta"]
            values[f"{var}_phi"] = components["phi"]
            if include_derived:
                values[f"{var}_mag"] = components["magnitude"]
        return values

    for var, coeffs in entry.items():
        field = FieldCoefficients(field_space, coeffs=coeffs)
        values[var] = transform.synthesize_scalar(field)

    conductance_coordinates = {"log_conductance_magnitude", "log_hall_to_pedersen_ratio"}
    if include_derived and key == "conductance" and conductance_coordinates <= set(values):
        values.update(
            evaluate_conductance_values(
                values["log_conductance_magnitude"], values["log_hall_to_pedersen_ratio"]
            )
        )

    return values


__all__ = ["evaluate_projected_input"]
