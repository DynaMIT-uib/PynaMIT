"""Input projection and storage helpers for simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from kompe import SphericalGrid
from kompe.math import get_array_module
from kompe.spherical_transform import SphericalTransform

from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.schema import INPUT_VARIABLES, WIND_FORCING_INPUTS

if TYPE_CHECKING:
    from pynamit.simulation.input_preparation import InputPreparation


class _InputProjector:
    """Validate, project, and store inputs during preparation."""

    def __init__(self, preparation: InputPreparation):
        self.preparation = preparation
        self._projection_transforms = {}

    def projection_transform(self, key: str) -> SphericalTransform:
        """Return the shared projection transform for one input."""
        basis = self.preparation.data.schema.input_field_spaces[key].basis
        if basis not in self._projection_transforms:
            self._projection_transforms[basis] = SphericalTransform(
                basis,
                self.preparation.model_grid,
                area_weighted=self.preparation.config.area_weighted_least_squares,
            )
        return self._projection_transforms[basis]

    def radial_current_from_FAC(self, FAC, *, lat=None, lon=None, theta=None, phi=None):
        """Convert signed field-parallel samples to radial current.

        Positive ``FAC`` is parallel to the background magnetic-field
        vector. An upward-positive convention must first be converted to
        that signed field-parallel convention.
        """
        input_grid = SphericalGrid(lat=lat, lon=lon, theta=theta, phi=phi)
        field = MagneticFieldEvaluation(
            self.preparation.main_field, input_grid, self.preparation.config.RI
        )
        return FAC * field.unit_br

    @staticmethod
    def require_complete_values(label: str, **values) -> None:
        """Require either all named values or none of them."""
        supplied = [name for name, value in values.items() if value is not None]
        if len(supplied) == len(values):
            return
        if supplied:
            missing = ", ".join(name for name, value in values.items() if value is None)
            raise ValueError(f"{label} are incomplete; missing {missing}.")
        names = ", ".join(values)
        raise TypeError(f"{label} require {names}.")

    @staticmethod
    def require_no_sample_values(label: str, **values) -> None:
        """Reject sample values when coefficient values are supplied."""
        supplied = [name for name, value in values.items() if value is not None]
        if supplied:
            raise ValueError(
                f"{label} cannot be combined with sample values: {', '.join(supplied)}."
            )

    @staticmethod
    def validate_only_coefficients(
        label: str, *, lat=None, lon=None, theta=None, phi=None, sqrt_weights=None, reg_lambda=None
    ) -> None:
        """Reject projection controls on direct coefficient inputs."""
        supplied = [
            name
            for name, value in {
                "lat": lat,
                "lon": lon,
                "theta": theta,
                "phi": phi,
                "sqrt_weights": sqrt_weights,
                "reg_lambda": reg_lambda,
            }.items()
            if value is not None
        ]
        if supplied:
            raise ValueError(
                f"{label} are already projected coefficients and cannot be combined "
                f"with {', '.join(supplied)}."
            )

    def require_no_exclusive_conflict(self, key: str) -> None:
        """Reject mutually exclusive input streams."""
        if key not in WIND_FORCING_INPUTS:
            return
        present = [
            other
            for other in sorted(WIND_FORCING_INPUTS - {key})
            if other in self.preparation.data.input_series.datasets
        ]
        if present:
            representations = ", ".join(repr(name) for name in sorted({key, *present}))
            raise ValueError(
                f"Wind-forcing representations {representations} are mutually "
                "exclusive; use only one."
            )

    @staticmethod
    def tangential_input_data(key: str, theta_component, phi_component) -> dict[str, Any]:
        """Return tangential input data with time before component."""
        xp = get_array_module(theta_component, phi_component)
        theta_rows = xp.atleast_2d(theta_component)
        phi_rows = xp.atleast_2d(phi_component)
        return {key: xp.stack((theta_rows, phi_rows), axis=1)}

    def set_scalar_input(
        self,
        key: str,
        *,
        samples: dict[str, Any],
        coefficients: dict[str, Any],
        sample_label: str,
        coefficient_label: str,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        tolerance=1e-15,
    ) -> None:
        """Validate, project, and store one scalar input stream."""
        variables = INPUT_VARIABLES[key]
        self._validate_variables(key, variables, samples, "samples")
        self._validate_variables(key, variables, coefficients, "coefficients")

        if any(value is not None for value in coefficients.values()):
            self.validate_only_coefficients(
                coefficient_label,
                lat=lat,
                lon=lon,
                theta=theta,
                phi=phi,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
            )
            self.require_complete_values(coefficient_label, **coefficients)
            self.require_no_sample_values(coefficient_label, **samples)
            input_data = {
                var: get_array_module(coefficients[var]).atleast_2d(coefficients[var])
                for var in variables
            }
            self.add_input_coefficients(key, input_data, time)
            return

        self.require_complete_values(sample_label, **samples)
        self._validate_projection_controls(key, sqrt_weights=sqrt_weights, reg_lambda=reg_lambda)
        input_data = {
            var: get_array_module(samples[var]).atleast_2d(samples[var]) for var in variables
        }
        self.project_and_add_input(
            key,
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            tolerance=tolerance,
        )

    def set_tangential_input(
        self,
        key: str,
        *,
        theta_component,
        phi_component,
        coefficients,
        sample_label: str,
        coefficient_label: str,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        tolerance=1e-15,
    ) -> None:
        """Validate, project, and store one tangential input stream."""
        self.require_no_exclusive_conflict(key)
        if coefficients is not None:
            self.validate_only_coefficients(
                coefficient_label,
                lat=lat,
                lon=lon,
                theta=theta,
                phi=phi,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
            )
            self.require_no_sample_values(
                coefficient_label, **{f"{key}_theta": theta_component, f"{key}_phi": phi_component}
            )
            field_space = self.preparation.data.schema.input_field_spaces[key]
            xp = get_array_module(coefficients)
            coefficient_rows = xp.asarray(coefficients)
            if coefficient_rows.shape == field_space.coefficient_shape:
                coefficient_rows = coefficient_rows.reshape((1, *field_space.coefficient_shape))
            elif (
                coefficient_rows.ndim != len(field_space.coefficient_shape) + 1
                or coefficient_rows.shape[1:] != field_space.coefficient_shape
            ):
                raise ValueError(
                    f"{coefficient_label} must have shape {field_space.coefficient_shape} "
                    f"for one time or (time, {', '.join(map(str, field_space.coefficient_shape))})."
                )
            self.add_input_coefficients(key, {key: coefficient_rows}, time)
            return

        self.require_complete_values(
            sample_label, **{f"{key}_theta": theta_component, f"{key}_phi": phi_component}
        )
        input_data = self.tangential_input_data(key, theta_component, phi_component)
        self.project_and_add_input(
            key,
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            tolerance=tolerance,
        )

    def project_tangential_samples(
        self,
        key: str,
        theta_component,
        phi_component,
        *,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        tolerance=1e-15,
    ):
        """Project tangential samples."""
        input_data = self.tangential_input_data(key, theta_component, phi_component)
        input_time = self.resolve_input_times(time, input_data)
        input_grid = SphericalGrid(lat=lat, lon=lon, theta=theta, phi=phi)
        coeff_rows = self.projection_transform(key).analyze_helmholtz_samples(
            input_data[key],
            input_grid=input_grid,
            analysis_basis=self.preparation.data.schema.input_projection_bases[key],
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            tolerance=tolerance,
        )
        self._validate_time_rows(key, key, coeff_rows, input_time)
        return input_time, coeff_rows

    def evaluate_Q_eff_from_neutral_wind(self, input_time, wind_coeff_rows):
        """Evaluate wind-equivalent Q_eff samples on the model grid."""
        response = self.preparation._require_response()
        grid = self.preparation.geometry.model_grid
        wind_representation = self.preparation.data.schema.input_field_spaces["u"].basis
        wind_synthesis = wind_representation.helmholtz_synthesis_operator(grid)
        Q_eff_values = []
        for time_value, wind_coeffs in zip(input_time, wind_coeff_rows, strict=True):
            response.activate_inputs_at_time(self.preparation.data.input_series, time_value)
            wind_on_grid = wind_synthesis.matvec(wind_coeffs).reshape((2, grid.size))
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

    def fit_Q_eff_from_neutral_wind(
        self, input_time, wind_coeff_rows, *, reg_lambda=None, tolerance=1e-15
    ):
        """Fit stored Q_eff coefficients to wind-driven E."""
        response = self.preparation._require_response()
        q_field_space = self.preparation.data.schema.input_field_spaces["Q_eff"]
        q_eff_synthesis_operator = q_field_space.basis.helmholtz_synthesis_operator(
            self.preparation.geometry.model_grid
        )
        q_coeff_rows = []
        cached_resistance_tensor = None
        solve_Q_eff = None
        for time_value, wind_coeffs in zip(input_time, wind_coeff_rows, strict=True):
            response.activate_inputs_at_time(self.preparation.data.input_series, time_value)
            E_wind_coeffs = response.u_coeffs_to_E_coeffs_operator.matvec(wind_coeffs)
            resistance_tensor = response.resistance_tensor_on_grid
            if resistance_tensor is not cached_resistance_tensor:
                cached_resistance_tensor = resistance_tensor
                Q_eff_to_E_operator = ionospheric_closure.tangential_current_to_E_coeffs_operator(
                    self.preparation.geometry.helmholtz_analysis_operator,
                    resistance_tensor,
                    q_eff_synthesis_operator,
                )
                solve_Q_eff = ionospheric_closure.build_Q_eff_coefficient_solver(
                    Q_eff_to_E_operator, reg_lambda=reg_lambda, tolerance=tolerance
                )
            q_coeffs = solve_Q_eff(E_wind_coeffs)
            q_coeff_rows.append(
                q_field_space.validate_coefficients(q_coeffs, name="Q_eff coefficients")
            )
        xp = get_array_module(*q_coeff_rows)
        return xp.stack(q_coeff_rows)

    def project_and_add_input(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        tolerance=1e-15,
    ) -> None:
        """Project gridded input data and store coefficient entries."""
        input_time = self.resolve_input_times(time, input_data)
        input_grid = SphericalGrid(lat=lat, lon=lon, theta=theta, phi=phi)
        transform = self.projection_transform(key)
        field_space = self.preparation.data.schema.input_field_spaces[key]
        if field_space.field_type == "scalar" and len(input_data) > 1:
            projected_data = self.project_scalar_input_variables(
                key,
                input_data,
                input_grid=input_grid,
                input_time=input_time,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                tolerance=tolerance,
            )
        else:
            projected_data = {}
            project = (
                transform.analyze_helmholtz_samples
                if field_space.field_type == "tangential"
                else transform.analyze_scalar_samples
            )

            for var, values in input_data.items():
                projected_values = project(
                    values,
                    input_grid=input_grid,
                    analysis_basis=self.preparation.data.schema.input_projection_bases[key],
                    sqrt_weights=sqrt_weights,
                    reg_lambda=reg_lambda,
                    tolerance=tolerance,
                )
                self._validate_time_rows(key, var, projected_values, input_time)
                projected_data[var] = projected_values

        self._store_input_rows(key, projected_data, input_time)

    def project_scalar_input_variables(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        input_grid,
        input_time,
        sqrt_weights=None,
        reg_lambda=None,
        tolerance=1e-15,
    ) -> dict[str, Any]:
        """Project scalar input variables in one batched transform."""
        transform = self.projection_transform(key)
        normalized = {
            var: transform.as_scalar_sample_rows(values, input_grid)
            for var, values in input_data.items()
        }
        for var, values in normalized.items():
            self._validate_time_rows(key, var, values, input_time)

        variables = tuple(normalized)
        xp = get_array_module(*(normalized[var] for var in variables))
        combined = xp.concatenate([normalized[var] for var in variables], axis=0)
        projected = transform.analyze_scalar_samples(
            combined,
            input_grid=input_grid,
            analysis_basis=self.preparation.data.schema.input_projection_bases[key],
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            tolerance=tolerance,
        )
        return {
            var: projected[index * input_time.size : (index + 1) * input_time.size]
            for index, var in enumerate(variables)
        }

    def _store_input_rows(self, key: str, projected_data: dict[str, Any], input_time) -> None:
        """Store and persist coefficient rows for one input."""
        for time_index in range(input_time.size):
            self.preparation.data.input_series.add_entry(
                key,
                {var: projected_data[var][time_index] for var in projected_data},
                input_time[time_index],
            )
        self.preparation.data.input_series.save(key, self.preparation.data.artifact_store)

    def add_input_coefficients(self, key: str, input_data: dict[str, Any], time) -> None:
        """Store input-basis coefficients directly in a time series."""
        input_time = self.resolve_input_times(time, input_data)
        self.validate_input_time_rows(key, input_time, input_data)
        self._store_input_rows(key, input_data, input_time)

    def resolve_input_times(self, time, data: dict[str, Any]):
        """Resolve times, defaulting one row to the current time."""
        if time is None:
            if any(data[var].shape[0] > 1 for var in data):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return np.atleast_1d(self.preparation.current_time)
        return np.atleast_1d(time)

    @staticmethod
    def validate_input_time_rows(key: str, input_time, input_data: dict[str, Any]) -> None:
        """Require input rows to match times."""
        row_counts = {var: int(values.shape[0]) for var, values in input_data.items()}
        if not row_counts:
            raise ValueError(f"{key} input data cannot be empty.")
        unique_counts = set(row_counts.values())
        if len(unique_counts) != 1:
            details = ", ".join(f"{var}={count}" for var, count in row_counts.items())
            raise ValueError(f"{key} input variables have inconsistent time rows: {details}.")
        row_count = next(iter(unique_counts))
        if row_count != int(input_time.size):
            details = ", ".join(f"{var}={count}" for var, count in row_counts.items())
            raise ValueError(
                f"{key} received {int(input_time.size)} time values, but input data has "
                f"{row_count} time rows ({details})."
            )

    @staticmethod
    def _validate_variables(
        key: str, variables: tuple[str, ...], values: dict[str, Any], value_kind: str
    ) -> None:
        """Require input dictionaries to match their declared spec."""
        expected = set(variables)
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"{key} {value_kind} must use variables {sorted(expected)}, got {sorted(actual)}."
            )

    @staticmethod
    def _validate_time_rows(key: str, var: str, values, input_time) -> None:
        """Require one data batch to match its times."""
        if values.shape[0] != input_time.size:
            raise ValueError(
                f"{key}.{var} has {values.shape[0]} data rows, but "
                f"{input_time.size} time values were supplied."
            )

    def _validate_projection_controls(self, key: str, *, sqrt_weights, reg_lambda) -> None:
        """Reject controls unsupported by CS storage."""
        projection_basis = getattr(self.preparation.config, f"{key}_projection_basis")
        if (
            key == "conductance"
            and projection_basis == "CS"
            and (sqrt_weights is not None or reg_lambda is not None)
        ):
            raise ValueError(
                f"sqrt_weights and reg_lambda are not supported for {key}_projection_basis='CS'."
            )
