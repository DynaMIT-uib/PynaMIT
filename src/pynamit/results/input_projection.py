"""Inspect projected simulation inputs on evaluation grids."""

from kompe.spherical_transform import SphericalTransform

from pynamit.fields import FieldCoefficients
from pynamit.results.field_maps import (
    evaluate_conductance_values,
    evaluate_tangential_coefficients,
)
from pynamit.results.grid_evaluation import transform_for_basis


def evaluate_projected_input(
    source, key, time, *, grid=None, transform=None, interpolation=False, include_derived=True
):
    """Evaluate one projected input time series on a grid.

    Parameters
    ----------
    source : Simulation or FieldTimeSeries
        Object containing projected input coefficient time series.
    key : str
        Input key, for example ``"boundary_jr"``, ``"boundary_Br"``,
        ``"conductance"``, ``"u"``, ``"Q_eff"``, or
        ``"E_neutral_wind"``.
    time : float
        Time value to select from the input time series.
    grid : SphericalGrid, optional
        Target grid. Required unless ``transform`` is supplied or
        ``source`` is a ``Simulation`` with a model geometry grid.
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
    series = source.data.input_series if hasattr(source, "data") else source
    entry = series.get_entry(key, time, interpolation=interpolation)
    if entry is None:
        raise ValueError(f"No {key!r} input is available at t={float(time):.3f}.")

    field_space = series.get_field_space(key)
    target_grid = grid
    if target_grid is None and hasattr(source, "geometry"):
        target_grid = source.geometry.model_grid
    if transform is not None:
        evaluator = transform_for_basis(field_space.representation, transform)
    elif target_grid is not None:
        evaluator = SphericalTransform(field_space.representation, target_grid)
    else:
        raise ValueError("A target grid or transform is required.")

    values = {}
    if field_space.field_type == "tangential":
        for var, coeffs in entry.items():
            field = FieldCoefficients(field_space, coeffs=coeffs)
            components = evaluate_tangential_coefficients(
                evaluator, field, include_magnitude=include_derived
            )
            values[f"{var}_theta"] = components["theta"]
            values[f"{var}_phi"] = components["phi"]
            if include_derived:
                values[f"{var}_mag"] = components["magnitude"]
        return values

    for var, coeffs in entry.items():
        field = FieldCoefficients(field_space, coeffs=coeffs)
        values[var] = evaluator.synthesize_scalar(field)

    conductance_coordinates = {"log_conductance_magnitude", "log_hall_to_pedersen_ratio"}
    if include_derived and key == "conductance" and conductance_coordinates <= set(values):
        values.update(
            evaluate_conductance_values(
                values["log_conductance_magnitude"], values["log_hall_to_pedersen_ratio"]
            )
        )

    return values


__all__ = ["evaluate_projected_input"]
