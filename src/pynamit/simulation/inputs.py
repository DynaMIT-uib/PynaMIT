"""Input projection and storage helpers for simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pynamit.simulation.schema import INPUT_FIELD_TYPES, INPUT_VARIABLES
from pynamit.sphere import Grid

WIND_SOURCE_GROUP = "wind_source"


@dataclass(frozen=True)
class InputSpec:
    """Declarative metadata for one simulation input stream."""

    key: str
    variables: tuple[str, ...]
    field_type: str
    exclusive_group: str | None = None
    reject_least_squares_for_cs_projection: bool = False

    @property
    def is_tangential(self) -> bool:
        """Return True for tangential inputs."""
        return self.field_type == "tangential"


def build_input_specs() -> dict[str, InputSpec]:
    """Return the canonical input-stream specifications."""
    specs = {}
    for key, variables in INPUT_VARIABLES.items():
        specs[key] = InputSpec(
            key=key,
            variables=tuple(variables),
            field_type=INPUT_FIELD_TYPES[key],
            exclusive_group=(WIND_SOURCE_GROUP if key in {"u", "Q_eff", "E_source"} else None),
            reject_least_squares_for_cs_projection=(key == "conductance"),
        )
    return specs


INPUT_SPECS = build_input_specs()


class InputProjector:
    """Project and store gridded or coefficient simulation inputs."""

    def __init__(self, owner: Any, specs: dict[str, InputSpec] | None = None):
        self.owner = owner
        self.specs = INPUT_SPECS if specs is None else dict(specs)

    def spec(self, key: str) -> InputSpec:
        """Return the specification for one input key."""
        try:
            return self.specs[key]
        except KeyError as exc:
            raise KeyError(f"Unknown simulation input {key!r}.") from exc

    @property
    def input_timeseries(self):
        """Return the owner's current input time series."""
        return self.owner.input_timeseries

    @property
    def input_transforms(self):
        """Return the owner's input transform map."""
        return self.owner.input_transforms

    @property
    def input_field_spaces(self):
        """Return the owner's input field-space map."""
        return self.owner.input_field_spaces

    @property
    def input_projection_bases(self):
        """Return the owner's input projection-basis map."""
        return self.owner.input_projection_bases

    def require_sample_values(self, label: str, **values) -> None:
        """Require all named sample values."""
        if any(value is None for value in values.values()):
            names = ", ".join(values)
            raise TypeError(f"{label} samples require {names}.")

    def require_complete_values(self, label: str, **values) -> None:
        """Require either all named values or none of them."""
        supplied = [name for name, value in values.items() if value is not None]
        if len(supplied) == len(values):
            return
        if supplied:
            missing = ", ".join(name for name, value in values.items() if value is None)
            raise ValueError(f"{label} are incomplete; missing {missing}.")
        names = ", ".join(values)
        raise TypeError(f"{label} require {names}.")

    def require_no_sample_values(self, label: str, **values) -> None:
        """Reject sample values when coefficient values are supplied."""
        supplied = [name for name, value in values.items() if value is not None]
        if supplied:
            raise ValueError(
                f"{label} cannot be combined with sample values: {', '.join(supplied)}."
            )

    def validate_only_coefficients(
        self,
        label: str,
        *,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        sqrt_weights=None,
        reg_lambda=None,
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
        spec = self.spec(key)
        if spec.exclusive_group is None:
            return
        group_keys = {
            item.key
            for item in self.specs.values()
            if item.exclusive_group == spec.exclusive_group and item.key != key
        }
        present = [
            other for other in sorted(group_keys) if other in self.input_timeseries.datasets
        ]
        if present:
            raise ValueError(
                "Neutral wind input 'u', effective-current input 'Q_eff', and "
                "direct electric-field input 'E_source' are mutually exclusive; "
                "use only one wind forcing representation."
            )

    def tangential_input_data(self, key: str, theta_component, phi_component) -> dict[str, Any]:
        """Return tangential input data with time before component."""
        data = {key: np.array([np.atleast_2d(theta_component), np.atleast_2d(phi_component)])}
        data[key] = np.moveaxis(data[key], [0, 1], [1, 0])
        return data

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
        spec = self.spec(key)
        self._validate_variables(spec, samples, "samples")
        self._validate_variables(spec, coefficients, "coefficients")

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
        input_time = self.adapt_input_time(time, input_data)
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        coeff_rows = self.input_transforms[key].project_helmholtz(
            input_data[key],
            input_grid=input_grid,
            projection_basis=self.input_projection_bases[key],
            sqrt_weights=sqrt_weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )
        self._validate_projected_time_rows(key, key, coeff_rows, input_time)
        return input_time, coeff_rows

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
        input_time = self.adapt_input_time(time, input_data)
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)
        transform = self.input_transforms[key]
        field_space = self.input_field_spaces[key]
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
                    projection_basis=self.input_projection_bases[key],
                    sqrt_weights=sqrt_weights,
                    reg_lambda=reg_lambda,
                    pinv_rtol=pinv_rtol,
                )
                self._validate_projected_time_rows(key, var, projected_values, input_time)
                projected_data[var] = projected_values

        self.add_projected_rows(key, projected_data, input_time)
        self.owner.data.save_input_dataset(key)

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
        transform = self.input_transforms[key]
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
            projection_basis=self.input_projection_bases[key],
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
            self.input_timeseries.add_entry(
                key,
                {var: projected_data[var][time_index] for var in projected_data},
                input_time[time_index],
            )

    def add_input_coefficients(self, key: str, input_data: dict[str, Any], time) -> None:
        """Store input-basis coefficients directly in a time series."""
        input_time = self.adapt_input_time(time, input_data)
        self.validate_input_time_rows(key, input_time, input_data)
        self.add_projected_rows(key, input_data, input_time)
        self.owner.data.save_input_dataset(key)

    def adapt_input_time(self, time, data: dict[str, Any]):
        """Return time values compatible with input data rows."""
        if time is None:
            if any(data[var].shape[0] > 1 for var in data):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return np.atleast_1d(self.owner.current_time)
        return np.atleast_1d(time)

    def validate_input_time_rows(self, key: str, input_time, input_data: dict[str, Any]) -> None:
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

    def _validate_variables(
        self, spec: InputSpec, values: dict[str, Any], value_kind: str
    ) -> None:
        """Require input dictionaries to match their declared spec."""
        expected = set(spec.variables)
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"{spec.key} {value_kind} must use variables {sorted(expected)}, "
                f"got {sorted(actual)}."
            )

    def _validate_projected_time_rows(self, key: str, var: str, values, input_time) -> None:
        """Require projected data rows to match supplied input times."""
        if values.shape[0] != input_time.size:
            raise ValueError(
                f"{key}.{var} has {values.shape[0]} projected time "
                f"slices, but {input_time.size} time values were supplied."
            )

    def _validate_projection_controls(self, key: str, *, sqrt_weights, reg_lambda) -> None:
        """Reject controls unsupported by CS storage."""
        spec = self.spec(key)
        projection_basis = getattr(self.owner.config, f"{key}_projection_basis", None)
        if (
            spec.reject_least_squares_for_cs_projection
            and projection_basis == "CS"
            and (sqrt_weights is not None or reg_lambda is not None)
        ):
            raise ValueError(
                f"sqrt_weights and reg_lambda are not supported for {key}_projection_basis='CS'."
            )


__all__ = ["INPUT_SPECS", "InputProjector", "InputSpec", "build_input_specs"]
