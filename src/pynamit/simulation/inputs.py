"""Input projection and storage helpers for simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pynamit.geomagnetism import MagneticFieldEvaluation
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.schema import INPUT_VARIABLES
from pynamit.sphere import Grid
from pynamit.sphere.spherical_transform import SphericalTransform

if TYPE_CHECKING:
    from pynamit.simulation.api import Simulation

_WIND_FORCING_GROUP = "wind_forcing"


@dataclass(frozen=True)
class _InputSpec:
    """Declarative metadata for one simulation input stream."""

    variables: tuple[str, ...]
    exclusive_group: str | None = None
    reject_least_squares_for_cs_projection: bool = False


_INPUT_SPECS = {
    key: _InputSpec(
        variables=tuple(variables),
        exclusive_group=_WIND_FORCING_GROUP if key in {"u", "Q_eff"} else None,
        reject_least_squares_for_cs_projection=(key == "resistance"),
    )
    for key, variables in INPUT_VARIABLES.items()
}


class InputPipeline:
    """Validate, project, and store simulation inputs."""

    def __init__(self, simulation: Simulation):
        self.simulation = simulation
        self._transforms_by_representation = {}
        self.projection_transforms = {}

    def projection_transform_for(self, key: str) -> SphericalTransform:
        """Return the shared projection transform for one input."""
        self._spec(key)
        if key not in self.projection_transforms:
            representation = self.simulation.run_data.schema.input_field_spaces[key].representation
            cache_key = getattr(
                representation,
                "signature",
                getattr(representation, "coefficient_space_signature", id(representation)),
            )
            if cache_key not in self._transforms_by_representation:
                self._transforms_by_representation[cache_key] = SphericalTransform(
                    representation,
                    self.simulation.geometry.model_grid,
                    grid_remap_basis=self.simulation.run_data.schema.cs_basis,
                    area_weighted=self.simulation.config.area_weighted_least_squares,
                )
            self.projection_transforms[key] = self._transforms_by_representation[cache_key]
        return self.projection_transforms[key]

    def radial_current_from_FAC(self, FAC, *, lat=None, lon=None, theta=None, phi=None):
        """Convert signed field-parallel samples to radial current.

        Positive ``FAC`` is parallel to the background magnetic-field
        vector. An upward-positive convention must first be converted to
        that signed field-parallel convention.
        """
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        field = MagneticFieldEvaluation(
            self.simulation.geometry.main_field, input_grid, self.simulation.config.RI
        )
        return FAC * field.unit_br

    @staticmethod
    def _spec(key: str) -> _InputSpec:
        """Return the specification for one input key."""
        try:
            return _INPUT_SPECS[key]
        except KeyError as exc:
            raise KeyError(f"Unknown simulation input {key!r}.") from exc

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
        spec = self._spec(key)
        if spec.exclusive_group is None:
            return
        group_keys = {
            other_key
            for other_key, item in _INPUT_SPECS.items()
            if item.exclusive_group == spec.exclusive_group and other_key != key
        }
        present = [
            other
            for other in sorted(group_keys)
            if other in self.simulation.run_data.input_series.datasets
        ]
        if present:
            raise ValueError(
                "Neutral wind input 'u' and effective-current input 'Q_eff' are "
                "mutually exclusive; use only one wind-forcing representation."
            )

    @staticmethod
    def tangential_input_data(key: str, theta_component, phi_component) -> dict[str, Any]:
        """Return tangential input data with time before component."""
        theta_rows = np.atleast_2d(theta_component)
        phi_rows = np.atleast_2d(phi_component)
        return {key: np.stack((theta_rows, phi_rows), axis=1)}

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
        pinv_rtol=1e-15,
    ) -> None:
        """Validate, project, and store one scalar input stream."""
        spec = self._spec(key)
        self._validate_variables(key, spec, samples, "samples")
        self._validate_variables(key, spec, coefficients, "coefficients")

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
            input_data = {var: np.atleast_2d(coefficients[var]) for var in spec.variables}
            self.add_input_coefficients(key, input_data, time)
            return

        self.require_complete_values(sample_label, **samples)
        self._validate_projection_controls(key, sqrt_weights=sqrt_weights, reg_lambda=reg_lambda)
        input_data = {var: np.atleast_2d(samples[var]) for var in spec.variables}
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
            pinv_rtol=pinv_rtol,
        )

    def set_tangential_input(
        self,
        key: str,
        *,
        theta_component,
        phi_component,
        cf_coefficients,
        df_coefficients,
        sample_label: str,
        coefficient_label: str,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ) -> None:
        """Validate, project, and store one tangential input stream."""
        self.require_no_exclusive_conflict(key)
        if cf_coefficients is not None or df_coefficients is not None:
            self.validate_only_coefficients(
                coefficient_label,
                lat=lat,
                lon=lon,
                theta=theta,
                phi=phi,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
            )
            self.require_complete_values(
                coefficient_label, **{f"{key}_cf": cf_coefficients, f"{key}_df": df_coefficients}
            )
            self.require_no_sample_values(
                coefficient_label, **{f"{key}_theta": theta_component, f"{key}_phi": phi_component}
            )
            input_data = self.tangential_input_data(key, cf_coefficients, df_coefficients)
            self.add_input_coefficients(key, input_data, time)
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
            pinv_rtol=pinv_rtol,
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
        pinv_rtol=1e-15,
    ):
        """Project tangential samples."""
        input_data = self.tangential_input_data(key, theta_component, phi_component)
        input_time = self.resolve_input_times(time, input_data)
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        coeff_rows = self.projection_transform_for(key).project_helmholtz(
            input_data[key],
            input_grid=input_grid,
            projection_basis=self.simulation.run_data.schema.input_projection_bases[key],
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        self._validate_projected_time_rows(key, key, coeff_rows, input_time)
        return input_time, coeff_rows

    def evaluate_Q_eff_from_neutral_wind(self, input_time, wind_coeff_rows):
        """Evaluate wind-equivalent Q_eff samples on the model grid."""
        response = self.simulation.response
        grid = self.simulation.geometry.model_grid
        wind_representation = self.simulation.run_data.schema.input_field_spaces[
            "u"
        ].representation
        wind_synthesis = wind_representation.get_helmholtz_synthesis_operator(grid)
        Q_eff_values = []
        for time_value, wind_coeffs in zip(input_time, wind_coeff_rows, strict=True):
            response.activate_inputs_at_time(self.simulation.run_data.input_series, time_value)
            wind_on_grid = np.asarray(wind_synthesis.matvec(wind_coeffs)).reshape((2, grid.size))
            Q_eff_values.append(
                ionospheric_closure.Q_eff_on_grid_from_wind(
                    wind_on_grid,
                    self.simulation.geometry.wind_motional_E_tensor,
                    response.resistance_tensor_on_grid,
                )
            )

        values = np.asarray(Q_eff_values)
        return values[:, 0, :], values[:, 1, :], grid.lat, grid.lon

    def fit_Q_eff_from_neutral_wind(
        self, input_time, wind_coeff_rows, *, reg_lambda=None, pinv_rtol=1e-15
    ):
        """Fit stored Q_eff coefficients to wind-driven E."""
        response = self.simulation.response
        q_field_space = self.simulation.run_data.schema.input_field_spaces["Q_eff"]
        q_synthesis = q_field_space.representation.get_helmholtz_synthesis_operator(
            self.simulation.geometry.model_grid
        )
        q_coeff_rows = []
        cached_resistance_tensor = None
        Q_eff_to_E = None
        for time_value, wind_coeffs in zip(input_time, wind_coeff_rows, strict=True):
            response.activate_inputs_at_time(self.simulation.run_data.input_series, time_value)
            E_wind_coeffs = response.u_coeffs_to_E_coeffs.matvec(wind_coeffs)
            resistance_tensor = response.resistance_tensor_on_grid
            if resistance_tensor is not cached_resistance_tensor:
                cached_resistance_tensor = resistance_tensor
                Q_eff_to_E = ionospheric_closure.tangential_current_to_E_coeffs_operator(
                    self.simulation.geometry.helmholtz_analysis_operator,
                    resistance_tensor,
                    q_synthesis,
                )
            q_coeffs = ionospheric_closure.solve_Q_eff_coefficients(
                Q_eff_to_E, E_wind_coeffs, reg_lambda=reg_lambda, pinv_rtol=pinv_rtol
            )
            q_coeff_rows.append(
                q_field_space.validate_coefficients(q_coeffs, name="Q_eff coefficients")
            )
        return np.asarray(q_coeff_rows)

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
        pinv_rtol=1e-15,
    ) -> None:
        """Project gridded input data and store coefficient entries."""
        input_time = self.resolve_input_times(time, input_data)
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        transform = self.projection_transform_for(key)
        field_space = self.simulation.run_data.schema.input_field_spaces[key]
        if field_space.field_type == "scalar" and len(input_data) > 1:
            projected_data = self.project_scalar_input_variables(
                key,
                input_data,
                input_grid=input_grid,
                input_time=input_time,
                sqrt_weights=sqrt_weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )
        else:
            projected_data = {}
            project = (
                transform.project_helmholtz
                if field_space.field_type == "tangential"
                else transform.project_scalar
            )

            for var, values in input_data.items():
                projected_values = project(
                    values,
                    input_grid=input_grid,
                    projection_basis=self.simulation.run_data.schema.input_projection_bases[key],
                    sqrt_weights=sqrt_weights,
                    reg_lambda=reg_lambda,
                    pinv_rtol=pinv_rtol,
                )
                self._validate_projected_time_rows(key, var, projected_values, input_time)
                projected_data[var] = projected_values

        self.add_projected_rows(key, projected_data, input_time)
        self.simulation.run_data.save_input_dataset(key)

    def project_scalar_input_variables(
        self,
        key: str,
        input_data: dict[str, Any],
        *,
        input_grid,
        input_time,
        sqrt_weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ) -> dict[str, Any]:
        """Project scalar input variables in one batched transform."""
        transform = self.projection_transform_for(key)
        normalized = {
            var: transform.normalize_scalar_value_batch(values, input_grid)
            for var, values in input_data.items()
        }
        for var, values in normalized.items():
            self._validate_projected_time_rows(key, var, values, input_time)

        variables = tuple(normalized)
        combined = np.concatenate([normalized[var] for var in variables], axis=0)
        projected = transform.project_scalar(
            combined,
            input_grid=input_grid,
            projection_basis=self.simulation.run_data.schema.input_projection_bases[key],
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        return {
            var: projected[index * input_time.size : (index + 1) * input_time.size]
            for index, var in enumerate(variables)
        }

    def add_projected_rows(self, key: str, projected_data: dict[str, Any], input_time) -> None:
        """Store projected coefficient rows."""
        for time_index in range(input_time.size):
            self.simulation.run_data.input_series.add_entry(
                key,
                {var: projected_data[var][time_index] for var in projected_data},
                input_time[time_index],
            )

    def add_input_coefficients(self, key: str, input_data: dict[str, Any], time) -> None:
        """Store input-basis coefficients directly in a time series."""
        input_time = self.resolve_input_times(time, input_data)
        self.validate_input_time_rows(key, input_time, input_data)
        self.add_projected_rows(key, input_data, input_time)
        self.simulation.run_data.save_input_dataset(key)

    def resolve_input_times(self, time, data: dict[str, Any]):
        """Resolve times, defaulting one row to the current time."""
        if time is None:
            if any(data[var].shape[0] > 1 for var in data):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return np.atleast_1d(self.simulation.current_time)
        return np.atleast_1d(time)

    @staticmethod
    def validate_input_time_rows(key: str, input_time, input_data: dict[str, Any]) -> None:
        """Require input rows to match times."""
        row_counts = {var: int(np.asarray(values).shape[0]) for var, values in input_data.items()}
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
        key: str, spec: _InputSpec, values: dict[str, Any], value_kind: str
    ) -> None:
        """Require input dictionaries to match their declared spec."""
        expected = set(spec.variables)
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"{key} {value_kind} must use variables {sorted(expected)}, got {sorted(actual)}."
            )

    @staticmethod
    def _validate_projected_time_rows(key: str, var: str, values, input_time) -> None:
        """Require projected data rows to match supplied input times."""
        if values.shape[0] != input_time.size:
            raise ValueError(
                f"{key}.{var} has {values.shape[0]} projected time "
                f"slices, but {input_time.size} time values were supplied."
            )

    def _validate_projection_controls(self, key: str, *, sqrt_weights, reg_lambda) -> None:
        """Reject controls unsupported by CS storage."""
        spec = self._spec(key)
        projection_basis = getattr(self.simulation.config, f"{key}_projection_basis", None)
        if (
            spec.reject_least_squares_for_cs_projection
            and projection_basis == "CS"
            and (sqrt_weights is not None or reg_lambda is not None)
        ):
            raise ValueError(
                f"sqrt_weights and reg_lambda are not supported for {key}_projection_basis='CS'."
            )
