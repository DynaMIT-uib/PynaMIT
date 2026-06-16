"""Inspect projected simulation inputs on plotting grids."""

from pynamit.primitives.field_coefficients import FieldCoefficients
from pynamit.sphere.spherical_transform import SphericalTransform
from pynamit.visualization.field_maps import (
    evaluate_conductance_values,
    evaluate_tangential_coefficients,
)
from pynamit.visualization.grid_evaluation import transform_for_source


def _input_timeseries(source):
    """Return input timeseries from a source object."""
    return getattr(source, "input_timeseries", source)


def _default_grid(source):
    """Return the state/model grid for a Dynamics-like object."""
    try:
        return source.state.geometry.grid
    except AttributeError:
        return None


def _make_transform(field_space, grid, transform):
    """Return a transform targeting ``grid`` for ``field_space``."""
    if transform is not None:
        return transform_for_source(field_space.representation, transform)
    if grid is None:
        raise ValueError("A target grid or transform is required.")
    return SphericalTransform(field_space.representation, grid)


def evaluate_projected_input(
    source, key, time, *, grid=None, transform=None, interpolation=False, include_derived=True
):
    """Evaluate one projected input time series on a grid.

    Parameters
    ----------
    source : Dynamics or Timeseries
        Object containing projected input coefficient time series.
    key : str
        Input key, for example ``"jr"``, ``"Br"``, ``"conductance"``,
        ``"u"``, or ``"Q_eff"``.
    time : float
        Time value to select from the input time series.
    grid : Grid, optional
        Target grid. Required unless ``transform`` is supplied or
        ``source`` is a ``Dynamics`` with a state geometry grid.
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
    timeseries = _input_timeseries(source)
    entry = timeseries.get_entry(key, time, interpolation=interpolation)
    if entry is None:
        raise ValueError(f"No {key!r} input is available at t={float(time):.3f}.")

    field_space = timeseries.get_field_space(key)
    target_grid = _default_grid(source) if grid is None else grid
    evaluator = _make_transform(field_space, target_grid, transform)

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

    if include_derived and key == "conductance" and {"etaP", "etaH"} <= set(values):
        values.update(evaluate_conductance_values(values["etaP"], values["etaH"]))

    return values


__all__ = ["evaluate_projected_input"]
