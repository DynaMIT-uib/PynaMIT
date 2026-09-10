"""Input projection and storage helpers for simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from kompe import SphericalGrid
from kompe.math import LeastSquaresSolver, get_array_module
from kompe.spherical_transform import SphericalTransform

from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.response import ElectrodynamicResponse
from pynamit.simulation.schema import WIND_FORCING_INPUTS

if TYPE_CHECKING:
    from pynamit.simulation.input_preparation import InputPreparation


def _scalar_sample_rows(values, grid):
    """Normalize provider/user samples to storage's leading time axis.

    Points may be flat or retain the coordinate shape, with time first
    or last. Flat, time-first layouts take precedence when ambiguous.
    Numerical analysis receives an explicit axis conversion below.
    """
    xp = get_array_module(values)
    array = xp.asarray(values)
    for point_shape in ((grid.size,), grid.shape):
        if array.shape == point_shape:
            return array.reshape(1, grid.size)
        if array.shape[1:] == point_shape:
            return array.reshape(array.shape[0], grid.size)
        if array.shape[:-1] == point_shape:
            return xp.moveaxis(array, -1, 0).reshape(array.shape[-1], grid.size)
    raise ValueError(
        "Samples must match the flat or shaped grid, optionally with one "
        f"leading or trailing time axis; got {array.shape} for grid shape {grid.shape}."
    )


class _InputProjector:
    """Validate, project, and store inputs during preparation."""

    def __init__(self, preparation: InputPreparation):
        self.preparation = preparation
        self.solver = LeastSquaresSolver(
            method=preparation.config.least_squares_solver,
            tolerance=preparation.config.least_squares_tolerance,
            preconditioner=preparation.config.least_squares_preconditioner,
        )

    def projection_transform(self, key: str) -> SphericalTransform:
        """Return the shared projection transform for one input."""
        basis = self.preparation.schema.input_field_spaces[key].basis
        return self.preparation.geometry.horizontal_transform.with_basis(basis)

    def require_no_exclusive_conflict(self, key: str) -> None:
        """Reject mutually exclusive input streams."""
        if key not in WIND_FORCING_INPUTS:
            return
        present = [
            other
            for other in sorted(WIND_FORCING_INPUTS - {key})
            if other in self.preparation.input_series.datasets
        ]
        if present:
            representations = ", ".join(repr(name) for name in sorted({key, *present}))
            raise ValueError(
                f"Wind-forcing representations {representations} are mutually "
                "exclusive; use only one."
            )

    @staticmethod
    def tangential_input_data(key: str, theta_component, phi_component, grid) -> dict[str, Any]:
        """Return tangential input data with time before component."""
        xp = get_array_module(theta_component, phi_component)
        theta_rows = _scalar_sample_rows(theta_component, grid)
        phi_rows = _scalar_sample_rows(phi_component, grid)
        return {key: xp.stack((theta_rows, phi_rows), axis=1)}

    def set_tangential_input(
        self,
        key,
        theta_component,
        phi_component,
        *,
        grid,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
    ):
        """Fit tangential samples in their input coefficient space."""
        data = self.tangential_input_data(key, theta_component, phi_component, grid)
        self.project_and_store_input(
            key, data, input_grid=grid, time=time, sqrt_weights=sqrt_weights, reg_lambda=reg_lambda
        )

    def project_tangential_samples(
        self,
        key: str,
        theta_component,
        phi_component,
        *,
        grid,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
    ):
        """Project tangential samples."""
        input_grid = grid
        input_data = self.tangential_input_data(key, theta_component, phi_component, input_grid)
        input_time = self.resolve_input_times(time, input_data)
        coeff_rows = self._project_rows(
            key,
            input_data[key],
            input_grid=input_grid,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )
        self._validate_time_rows(key, key, coeff_rows, input_time)
        return input_time, coeff_rows

    def evaluate_Q_eff_from_neutral_wind(self, input_time, wind_coeff_rows):
        """Evaluate wind-equivalent Q_eff samples on the model grid."""
        response = None
        grid = self.preparation.geometry.model_grid
        wind_representation = self.preparation.schema.input_field_spaces["u"].basis
        wind_synthesis = wind_representation.helmholtz_synthesis_operator(grid)
        Q_eff_values = []
        series = self.preparation.input_series
        for time_value, wind_coeffs in zip(input_time, wind_coeff_rows, strict=True):
            response = ElectrodynamicResponse.from_conductance(
                self.preparation.geometry,
                self.preparation.config,
                series.get_field_space("conductance"),
                series.get_entry("conductance", time_value),
                previous=response,
            )
            wind_on_grid = wind_synthesis(wind_coeffs)
            Q_eff_values.append(
                ionospheric_closure.Q_eff_on_grid_from_wind(
                    wind_on_grid,
                    self.preparation.geometry.wind_motional_E_tensor,
                    response.resistance_tensor_on_grid,
                )
            )

        xp = get_array_module(*Q_eff_values)
        values = xp.stack(Q_eff_values)
        return values[:, 0, :], values[:, 1, :], grid.lat, grid.lon

    def fit_Q_eff_from_neutral_wind(self, input_time, wind_coeff_rows, *, reg_lambda=None):
        """Fit stored Q_eff coefficients to wind-driven E."""
        response = None
        q_field_space = self.preparation.schema.input_field_spaces["Q_eff"]
        q_eff_synthesis_operator = q_field_space.basis.helmholtz_synthesis_operator(
            self.preparation.geometry.model_grid
        )
        q_coeff_rows = []
        cached_resistance_tensor = None
        solve_Q_eff = None
        series = self.preparation.input_series
        for time_value, wind_coeffs in zip(input_time, wind_coeff_rows, strict=True):
            response = ElectrodynamicResponse.from_conductance(
                self.preparation.geometry,
                self.preparation.config,
                series.get_field_space("conductance"),
                series.get_entry("conductance", time_value),
                previous=response,
            )
            E_wind_coeffs = self.preparation.geometry.u_coeffs_to_E_coeffs_operator(wind_coeffs)
            resistance_tensor = response.resistance_tensor_on_grid
            if resistance_tensor is not cached_resistance_tensor:
                cached_resistance_tensor = resistance_tensor
                Q_eff_to_E_operator = ionospheric_closure.tangential_current_to_E_coeffs_operator(
                    self.preparation.geometry.helmholtz_analysis_operator,
                    resistance_tensor,
                    q_eff_synthesis_operator,
                )
                solve_Q_eff = ionospheric_closure.build_Q_eff_coefficient_solver(
                    Q_eff_to_E_operator,
                    solver=self.solver,
                    reg_lambda=reg_lambda,
                    gauge_constraints=q_field_space.basis.helmholtz_gauge_constraints,
                )
            q_coeffs = solve_Q_eff(E_wind_coeffs)
            q_coeff_rows.append(
                q_field_space.validate_coefficients(q_coeffs, name="Q_eff coefficients")
            )
        xp = get_array_module(*q_coeff_rows)
        return xp.stack(q_coeff_rows)

    def _project_rows(self, key, rows, *, input_grid, sqrt_weights=None, reg_lambda=None):
        """Remap and fit samples, returning coefficient time rows."""
        transform = self.projection_transform(key)
        schema = self.preparation.schema
        config = self.preparation.config
        mapping = (
            config.conductance_basis
            if key == "conductance"
            else getattr(config, f"{key}_remapping")
        )
        remapper = self.preparation.geometry.cs_basis.remapper if mapping == "CS" else None
        remap = None
        if schema.input_field_spaces[key].representation == "helmholtz":
            analyze = transform.analyze_helmholtz_samples
            if remapper is not None:
                remap = remapper.tangential_operator(input_grid, transform.grid)
        else:
            analyze = transform.analyze_scalar_samples
            if remapper is not None:
                remap = remapper.scalar_operator(input_grid, transform.grid)

        # Storage is time-first; Kompe keeps scientific axes first.
        xp = get_array_module(rows)
        coefficients = analyze(
            xp.moveaxis(rows, 0, -1),
            input_grid=input_grid,
            remap=remap,
            solver=self.solver,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
        )
        return get_array_module(coefficients).moveaxis(coefficients, -1, 0)

    def project_and_store_input(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        input_grid,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
    ) -> None:
        """Project gridded input data and store coefficient entries."""
        if not isinstance(input_grid, SphericalGrid):
            raise TypeError("grid must be a SphericalGrid in the simulation model frame.")
        self.require_no_exclusive_conflict(key)
        if key == "boundary_Br" and self.preparation.config.RM is None:
            raise ValueError("boundary_Br requires a finite magnetospheric radius RM.")
        if (
            key == "conductance"
            and self.preparation.config.conductance_basis == "CS"
            and (sqrt_weights is not None or reg_lambda is not None)
        ):
            raise ValueError(
                "CS conductance stores nodal values; fitting controls are not applicable."
            )
        field_space = self.preparation.schema.input_field_spaces[key]
        # All variables share this grid and coefficient space. Analyze
        # their time rows together, retaining scientific component axes.
        # Tangential setters have already stacked normalized components.
        normalized = (
            input_data
            if field_space.representation == "helmholtz"
            else {
                var: _scalar_sample_rows(values, input_grid) for var, values in input_data.items()
            }
        )
        input_time = self.resolve_input_times(time, normalized)
        for var, values in normalized.items():
            self._validate_time_rows(key, var, values, input_time)
        xp = get_array_module(*normalized.values())
        batches = list(normalized.values())
        samples = batches[0] if len(batches) == 1 else xp.concatenate(batches, axis=0)
        projected = self._project_rows(
            key, samples, input_grid=input_grid, sqrt_weights=sqrt_weights, reg_lambda=reg_lambda
        )
        projected_data = {
            var: projected[index * input_time.size : (index + 1) * input_time.size]
            for index, var in enumerate(normalized)
        }
        self._store_input_rows(key, projected_data, input_time)

    def _store_input_rows(self, key: str, projected_data: dict[str, Any], input_time) -> None:
        """Store and persist coefficient rows for one input."""
        self.preparation.input_series.add_entries(key, projected_data, input_time)
        if self.preparation.artifact_store is not None:
            self.preparation.input_series.save(key, self.preparation.artifact_store)

    def store_input_coefficients(self, key: str, input_data: dict[str, Any], time) -> None:
        """Store input-basis coefficients directly in a time series."""
        self.require_no_exclusive_conflict(key)
        if key == "boundary_Br" and self.preparation.config.RM is None:
            raise ValueError("boundary_Br requires a finite magnetospheric radius RM.")
        input_time = self.resolve_input_times(time, input_data)
        for var, values in input_data.items():
            self._validate_time_rows(key, var, values, input_time)
        self._store_input_rows(key, input_data, input_time)

    def resolve_input_times(self, time, data: dict[str, Any]):
        """Resolve explicit times, defaulting one row to time zero."""
        if time is None:
            if any(data[var].shape[0] > 1 for var in data):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return np.array([0.0])
        return np.atleast_1d(time)

    @staticmethod
    def _validate_time_rows(key: str, var: str, values, input_time) -> None:
        """Require one data batch to match its times."""
        if values.shape[0] != input_time.size:
            raise ValueError(
                f"{key}.{var} has {values.shape[0]} data rows, but "
                f"{input_time.size} time values were supplied."
            )
