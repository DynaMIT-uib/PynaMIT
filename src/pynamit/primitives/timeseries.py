"""Time-indexed coefficient storage.

``Timeseries`` stores named xarray datasets together with their coefficient
schemas. Each series key is described by a ``FieldSpec`` so the persistence
layer knows both the basis family and whether SH storage is mean-free.
"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pynamit.primitives.field_spec import FieldSpec

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class Timeseries:
    """Time-indexed container for coefficient datasets."""

    def __init__(self, storage_specs, variables):
        """Initialize one timeseries schema collection.

        Parameters
        ----------
        storage_specs : dict
            Mapping from series key to ``FieldSpec`` descriptors.
        variables : dict
            Mapping from series key to stored variable names and their field
            types.
        """
        self.variables = variables
        self.storage_specs = {
            key: self._normalize_storage_spec(key, spec)
            for key, spec in storage_specs.items()
        }

        self.datasets = {}

        self.basis_multiindices = {}
        for key in self.variables.keys():
            field_type = self.storage_specs[key].field_type
            if field_type == "scalar":
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    self._get_storage_index_arrays(key), names=self._get_storage_index_names(key)
                )
            elif field_type == "tangential":
                index_arrays = self._get_storage_index_arrays(key)
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    [
                        np.tile(index_arrays[i], 2)
                        for i in range(len(index_arrays))
                    ],
                    names=self._get_storage_index_names(key),
                )
            else:
                raise ValueError(
                    "Mixed scalar and tangential input (unsupported), or invalid input type"
                )

    def _infer_field_type(self, key: str) -> str:
        """Infer the common field type for one stored series key."""
        field_types = set(self.variables[key].values())
        if len(field_types) != 1:
            raise ValueError(
                "Mixed scalar and tangential input (unsupported), or invalid input type"
            )
        return str(next(iter(field_types)))

    def _normalize_storage_spec(self, key: str, spec: Any) -> FieldSpec:
        """Validate one storage descriptor."""
        field_type = self._infer_field_type(key)
        if isinstance(spec, FieldSpec):
            if spec.field_type != field_type:
                raise ValueError(
                    f"Storage spec for {key!r} declares field_type={spec.field_type!r}, "
                    f"but variables require {field_type!r}."
                )
            return spec
        raise TypeError(
            f"Storage spec for {key!r} must be a FieldSpec, got {type(spec).__name__}."
        )

    def _get_storage_index_arrays(self, key: str) -> list[np.ndarray]:
        """Return coefficient multiindex arrays for one stored series."""
        return list(self.storage_specs[key].index_arrays)

    def _get_storage_index_names(self, key: str) -> list[str]:
        """Return coordinate names for one stored series."""
        return list(self.storage_specs[key].basis.index_names)

    def get_storage_spec(self, key: str) -> FieldSpec:
        """Return the ``FieldSpec`` for one stored series."""
        return self.storage_specs[key]

    def load_all(self, io):
        """Load all timeseries from NetCDF files."""
        for key in self.variables.keys():
            self.load(key, io)

    def load(self, key, io):
        """Load a timeseries from NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to load.
        """
        dataset = io.load_dataset(key)

        if dataset is not None:
            basis_multiindex = pd.MultiIndex.from_arrays(
                [
                    dataset[self._get_storage_index_names(key)[i]].values
                    for i in range(len(self._get_storage_index_names(key)))
                ],
                names=self._get_storage_index_names(key),
            )
            coords = xr.Coordinates.from_pandas_multiindex(basis_multiindex, dim="i").merge(
                {"time": dataset.time.values}
            )
            self.datasets[key] = dataset.drop_vars(
                self._get_storage_index_names(key)
            ).assign_coords(coords)

    def get_data_var_name(self, key, var):
        """Return the stored xarray variable name for one timeseries entry."""
        return f"{self.get_storage_spec(key).kind}_{var}"

    def _build_entry_dataset(self, key: str, data: dict[str, np.ndarray], time: float) -> xr.Dataset:
        """Build a one-sample dataset for appending/replacing a stored entry."""
        data_vars = {}
        for var in data:
            data_vars[self.get_data_var_name(key, var)] = (
                ["time", "i"],
                np.asarray(data[var]).reshape((1, -1)),
            )

        coords = xr.Coordinates.from_pandas_multiindex(
            self.basis_multiindices[key],
            dim="i",
        ).merge({"time": [time]})
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def add_entry(self, key, data, time):
        """Add one time slice to a stored series.

        Parameters
        ----------
        key : {'jr', 'conductance', 'u', 'state', 'steady_state'}
            Series key.
        data : dict
            Mapping from stored variable name to coefficient values.
        time : float
            Time coordinate for the stored slice.
        """
        dataset = self._build_entry_dataset(key, data, time)

        if key not in self.datasets:
            self.datasets[key] = dataset
            return

        existing = self.datasets[key]
        time_value = float(time)
        time_coords = np.asarray(existing.time.values, dtype=float)

        if time_coords.size == 0:
            self.datasets[key] = dataset
            return

        # Common case: strictly append in chronological order. This still copies
        # the underlying xarray arrays, but avoids an unnecessary drop/sort pass.
        if time_value > float(time_coords[-1]) + FLOAT_ERROR_MARGIN:
            self.datasets[key] = xr.concat([existing, dataset], dim="time")
            return

        # Replace existing sample at the same timestamp or keep chronological order
        # for out-of-order insertion.
        self.datasets[key] = xr.concat(
            [existing.drop_sel(time=dataset.time, errors="ignore"), dataset],
            dim="time",
        ).sortby("time")

    def get_entry(self, key, time, interpolation=False):
        """Select, and optionally interpolate, time series data.

        Returns
        -------
        dict or None
            Latest data for the specified key, or ``None`` if no data is
            available.
        """
        return self.get_entry_with_derivative(key, time, interpolation=interpolation)[0]

    def get_entry_with_derivative(self, key, time, interpolation=False):
        """Select time series data and derivative corresponding to the specified time."""

        if np.any(self.datasets[key].time.values <= time + FLOAT_ERROR_MARGIN):
            current_data = {}
            current_derivative = {}

            # Select latest data before the current time.
            dataset_before = self.datasets[key].sel(
                time=[time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.variables[key]:
                current_data[var] = dataset_before[self.get_data_var_name(key, var)].values.flatten()
                # Default derivative is zero if no next point
                current_derivative[var] = np.zeros_like(current_data[var])

            # If requested, add linear interpolation correction.
            if interpolation and np.any(
                self.datasets[key].time.values > time + FLOAT_ERROR_MARGIN
            ):
                dataset_after = self.datasets[key].sel(
                    time=[time + FLOAT_ERROR_MARGIN], method="bfill"
                )
                
                dt = float(dataset_after.time.item() - dataset_before.time.item())
                if dt > 0:
                     factor = (time - dataset_before.time.item()) / dt
                     for var in self.variables[key]:
                        y0 = dataset_before[self.get_data_var_name(key, var)].values.flatten()
                        y1 = dataset_after[self.get_data_var_name(key, var)].values.flatten()
                        
                        slope = (y1 - y0) / dt
                        
                        current_data[var] += factor * (y1 - y0) # Optimized interp
                        current_derivative[var] = slope

            return current_data, current_derivative
        else:
            # No data available for the specified time.
            return None, None

    def save(self, key, io):
        io.save_dataset(self.datasets[key].reset_index("i"), key)
