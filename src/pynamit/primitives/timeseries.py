"""Timeseries Class.

This module contains the Timeseries class, which is responsible for
handling input and output operations in the simulation. It manages
the reading and writing of datasets, including time series data,
and provides methods for setting input data and selecting data for
the simulation.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pynamit.primitives.coefficient_field import CoefficientField
from pynamit.primitives.field_space import FieldSpace

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class Timeseries:
    """Timeseries Class.

    Class for handling input and output operations in the simulation.
    This class manages the reading and writing of datasets, including
    time series data, and provides methods for setting input data and
    selecting data for the simulation.
    """

    def __init__(
        self,
        field_spaces_or_cs_basis,
        storage_bases_or_vars,
        vars=None,
        area_weighted_least_squares=False,
    ):
        """Initialize the Timeseries class.

        Parameters
        ----------
        field_spaces_or_cs_basis : dict or CSBasis
            Mapping from time-series group to ``FieldSpace``. The legacy
            constructor form passes the cubed-sphere basis here.
        storage_bases_or_vars : dict
            Variable schema in the new form, or storage bases in the
            legacy constructor form.
        vars : dict
            Variable names and field types for each group when using the
            legacy constructor form.
        area_weighted_least_squares : bool, optional
            Preserved for callers that also construct projectors; the
            time-series object itself stores coefficients only.
        """
        if vars is None:
            self.cs_basis = None
            self.vars = storage_bases_or_vars
            self.field_spaces = {
                key: self._normalize_field_space(key, field_space)
                for key, field_space in field_spaces_or_cs_basis.items()
            }
        else:
            self.cs_basis = field_spaces_or_cs_basis
            self.vars = vars
            self.field_spaces = {
                key: FieldSpace.from_basis(
                    basis,
                    field_type=self._common_field_type(key),
                    mean_free=getattr(basis, "mean_free", False),
                )
                for key, basis in storage_bases_or_vars.items()
            }
        self.variables = self.vars

        self.area_weighted_least_squares = bool(area_weighted_least_squares)

        # Initialize variables and timeseries storage
        self.datasets = {}
        self.previous_data = {}
        self._pending_start: dict[str, int] = {}
        self._full_save_required: dict[str, bool] = {}
        self._storage_kinds: dict[str, str] = {}

        self.basis_multiindices = {}
        for key in self.vars.keys():
            self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                self.field_spaces[key].multiindex_arrays(),
                names=self.field_spaces[key].index_names,
            )

    @property
    def storage_bases(self):
        """Return storage bases derived from the canonical field spaces."""
        return {key: field_space.basis for key, field_space in self.field_spaces.items()}

    def get_storage_basis(self, key):
        """Return the storage basis for one stored series."""
        return self.field_spaces[key].basis

    def _common_field_type(self, key):
        """Return the shared field type for one time-series group."""
        field_types = {self.vars[key][var] for var in self.vars[key]}
        if len(field_types) != 1:
            raise ValueError(
                "Mixed scalar and tangential input (unsupported), or invalid input type"
            )
        return field_types.pop()

    def _normalize_field_space(self, key, field_space):
        """Validate or construct one field-space descriptor."""
        common_field_type = self._common_field_type(key)
        return FieldSpace.from_basis(field_space, field_type=common_field_type)

    def get_storage_spec(self, key):
        """Return the field-space descriptor for one stored series."""
        return self.field_spaces[key]

    def get_data_var_name(self, key, var):
        """Return the stored xarray variable name for one series variable."""
        return f"{self.get_storage_spec(key).kind}_{var}"

    def load_all(self, io):
        """Load all persisted timeseries datasets."""
        for key in self.vars.keys():
            self.load(key, io)

    def load(self, key, io):
        """Load a persisted timeseries dataset.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to load.
        """
        storage_kind = io.get_dataset_storage_kind(key)
        dataset = io.load_dataset(key)

        if dataset is not None:
            storage_basis = self.get_storage_basis(key)
            basis_multiindex = pd.MultiIndex.from_arrays(
                [
                    dataset[storage_basis.index_names[i]].values
                    for i in range(len(storage_basis.index_names))
                ],
                names=storage_basis.index_names,
            )
            coords = xr.Coordinates.from_pandas_multiindex(basis_multiindex, dim="i").merge(
                {"time": dataset.time.values}
            )
            self.datasets[key] = dataset.drop_vars(
                storage_basis.index_names
            ).assign_coords(coords)
            self._pending_start[key] = int(self.datasets[key].sizes.get("time", 0))
            self._full_save_required[key] = False
            if storage_kind is not None:
                self._storage_kinds[key] = storage_kind

    def add_entry(self, key, data, time):
        """Add entry to the timeseries.

        Creates a new timeseries if one does not exist, otherwise
        concatenates the new data along the time dimension.

        Parameters
        ----------
        key : {'jr', 'conductance', 'u', 'state', 'steady_state'}
            The key identifying the type of data.
        data : dict
            Dictionary of variables to set.
        time : float
            The time point for the data.
        """
        data_vars = {}
        for var in data:
            values = CoefficientField(
                self.field_spaces[key],
                data[var],
                name=f"{key}.{var}",
            ).coeffs
            data_vars[self.get_data_var_name(key, var)] = (
                ["time", "i"],
                values.reshape((1, -1)),
            )

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates.from_pandas_multiindex(
                self.basis_multiindices[key], dim="i"
            ).merge({"time": [time]}),
        )

        if key not in self.datasets:
            self.datasets[key] = dataset.sortby("time")
            self._pending_start[key] = 0
            self._full_save_required[key] = False
        else:
            existing = self.datasets[key]
            time_value = float(time)
            time_coords = np.asarray(existing.time.values, dtype=float)

            if time_coords.size == 0:
                self.datasets[key] = dataset.sortby("time")
                self._pending_start[key] = 0
                self._full_save_required[key] = False
                return

            if time_value > float(time_coords[-1]) + FLOAT_ERROR_MARGIN:
                previous_size = int(existing.sizes.get("time", 0))
                self.datasets[key] = xr.concat([existing, dataset], dim="time")
                pending_start = self._pending_start.get(key, previous_size)
                self._pending_start[key] = min(pending_start, previous_size)
                self._full_save_required[key] = bool(self._full_save_required.get(key, False))
                return

            self.datasets[key] = xr.concat(
                [existing.drop_sel(time=dataset.time, errors="ignore"), dataset], dim="time"
            ).sortby("time")
            self._pending_start[key] = 0
            self._full_save_required[key] = True

    def get_entry_if_changed(self, key, time, interpolation=False):
        """Select time series data corresponding to the specified time.

        Parameters
        ----------
        key : str
            Key for the time series.
        time : float
            Current time for which to select data.
        interpolation : bool, optional
            Whether to use linear interpolation.

        Returns
        -------
        dict or None
            Dictionary containing the latest data for the specified
            key, or None if no new data is available.
        """
        current_data = self.get_entry(key, time, interpolation=interpolation)

        if current_data is not None:
            # Check if the data has changed since the last time.
            if not all([var in self.previous_data.keys() for var in self.vars[key]]) or (
                not all(
                    [
                        np.allclose(
                            current_data[var],
                            self.previous_data[var],
                            rtol=FLOAT_ERROR_MARGIN,
                            atol=0.0,
                        )
                        for var in self.vars[key]
                    ]
                )
            ):
                # Update the previous data with the current data.
                for var in self.vars[key]:
                    self.previous_data[var] = current_data[var]

                return current_data

        # No new data available.
        return None

    def get_entry(self, key, time, interpolation=False):
        """Select time series data corresponding to the specified time.

        Parameters
        ----------
        key : str
            Key for the time series.
        time : float
            Current time for which to select data.
        interpolation : bool, optional
            Whether to use linear interpolation.

        Returns
        -------
        dict or None
            Dictionary containing the latest data for the specified
            key, or None if no data is available.
        """
        if np.any(self.datasets[key].time.values <= time + FLOAT_ERROR_MARGIN):
            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.datasets[key].sel(
                time=[time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.vars[key]:
                current_data[var] = dataset_before[
                    self.get_data_var_name(key, var)
                ].values.flatten()

            # If requested, add linear interpolation correction.
            if interpolation and np.any(
                self.datasets[key].time.values > time + FLOAT_ERROR_MARGIN
            ):
                dataset_after = self.datasets[key].sel(
                    time=[time + FLOAT_ERROR_MARGIN], method="bfill"
                )
                for var in self.vars[key]:
                    current_data[var] += (
                        (time - dataset_before.time.item())
                        / (dataset_after.time.item() - dataset_before.time.item())
                        * (
                            dataset_after[
                                self.get_data_var_name(key, var)
                            ].values.flatten()
                            - dataset_before[
                                self.get_data_var_name(key, var)
                            ].values.flatten()
                        )
                    )

            return current_data
        else:
            # No data available for the specified time.
            return None

    def save(self, key, io, *, print_info: bool = False):
        """Persist one stored series to disk.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to save.
        """
        dataset = self.datasets[key].reset_index("i")
        time_size = int(dataset.sizes.get("time", 0))
        pending_start = int(self._pending_start.get(key, 0))
        full_save_required = bool(self._full_save_required.get(key, False))
        existing_storage_kind = io.get_dataset_storage_kind(key)
        target_storage_kind = self._storage_kinds.get(key)

        if target_storage_kind is None:
            target_storage_kind = (
                existing_storage_kind
                if existing_storage_kind is not None
                else io.default_dataset_storage_kind(key)
            )

        if (
            existing_storage_kind is not None
            and not full_save_required
            and pending_start >= time_size
        ):
            self._storage_kinds[key] = existing_storage_kind
            return

        if (
            target_storage_kind == "zarr"
            and existing_storage_kind == "zarr"
            and not full_save_required
            and 0 < pending_start < time_size
        ):
            dataset_to_save = dataset.isel(time=slice(pending_start, None))
            io.save_dataset(
                dataset_to_save, key, print_info=print_info, storage="zarr", append_dim="time"
            )
        else:
            io.save_dataset(dataset, key, print_info=print_info, storage=target_storage_kind)

        self._pending_start[key] = time_size
        self._full_save_required[key] = False
        actual_storage_kind = io.get_dataset_storage_kind(key)
        self._storage_kinds[key] = (
            target_storage_kind if actual_storage_kind is None else actual_storage_kind
        )

    def trim_in_memory(self, key: str, *, keep_last: int) -> None:
        """Drop older samples while preserving append bookkeeping."""
        if keep_last < 0:
            raise ValueError(f"keep_last must be non-negative, got {keep_last!r}.")
        if key not in self.datasets:
            return

        dataset = self.datasets[key]
        time_size = int(dataset.sizes.get("time", 0))
        if time_size == 0 or keep_last >= time_size:
            return

        trimmed = dataset.isel(time=slice(time_size - keep_last, None))
        self.datasets[key] = trimmed
        trimmed_size = int(trimmed.sizes.get("time", 0))
        self._pending_start[key] = min(int(self._pending_start.get(key, 0)), trimmed_size)
