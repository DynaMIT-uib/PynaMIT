"""Time-series storage for field coefficients."""

import numpy as np
import pandas as pd
import xarray as xr
from pynamit.fields import FieldCoefficients
from pynamit.fields import FieldSpace

TIME_TOLERANCE_SECONDS = 1e-6
_VALUE_CHANGE_RTOL = 1e-6


class FieldTimeSeries:
    """Persist and select time-indexed field coefficients."""

    def __init__(self, field_spaces, variables):
        """Initialize named coefficient series from their field spaces.

        Parameters
        ----------
        field_spaces : dict
            Mapping from time-series group to ``FieldSpace``.
        variables : dict
            Variable names for each group.
        """
        self.variables = self._normalize_variables(variables)
        self.field_spaces = self._normalize_field_spaces(field_spaces)

        # Initialize in-memory series and persistence bookkeeping.
        self.datasets = {}
        self._previous_entries = {}
        self._pending_start: dict[str, int] = {}
        self._full_save_required: dict[str, bool] = {}
        self._storage_kinds: dict[str, str] = {}

        self._coefficient_indexes = {}
        for key in self.variables.keys():
            self._coefficient_indexes[key] = pd.MultiIndex.from_arrays(
                self.field_spaces[key].multiindex_arrays(),
                names=self.field_spaces[key].index_names,
            )

    def _normalize_variables(self, variables):
        """Return variable-name tuples after schema validation."""
        normalized = {}
        for key, names in variables.items():
            if isinstance(names, dict):
                raise TypeError(
                    "FieldTimeSeries variables must be sequences of variable names; "
                    "field types belong in FieldSpace."
                )
            if isinstance(names, str):
                raise TypeError("FieldTimeSeries variable groups must be sequences, not strings.")
            normalized[key] = tuple(names)
        return normalized

    def _normalize_field_spaces(self, field_spaces):
        """Return field spaces after schema validation."""
        if set(field_spaces) != set(self.variables):
            raise ValueError("FieldTimeSeries field_spaces and variables must use the same keys.")
        normalized = {}
        for key, field_space in field_spaces.items():
            if not isinstance(field_space, FieldSpace):
                raise TypeError(
                    "FieldTimeSeries field_spaces values must be FieldSpace instances."
                )
            normalized[key] = field_space
        return normalized

    def get_field_space(self, key):
        """Return the field-space descriptor for one stored series."""
        return self.field_spaces[key]

    def get_data_var_name(self, key, var):
        """Return the xarray variable name for one series variable."""
        return f"{self.get_field_space(key).kind}_{var}"

    @staticmethod
    def _time_value(time):
        """Return one finite scalar simulation time."""
        if isinstance(time, (bool, np.bool_)):
            raise ValueError("Time-series entries require a numeric time value.")
        if np.ndim(time) != 0:
            raise ValueError("Time-series entries require one scalar time value.")
        try:
            value = float(time)
        except (TypeError, ValueError) as exc:
            raise ValueError("Time-series entries require a numeric time value.") from exc
        if not np.isfinite(value):
            raise ValueError("Time-series entries require a finite time value.")
        return value

    def load_all(self, store):
        """Load all persisted time-series datasets."""
        for key in self.variables.keys():
            self.load(key, store)

    def _validate_loaded_dataset(self, key, dataset):
        """Validate persisted coefficients against their schema."""
        field_space = self.get_field_space(key)
        expected_data_vars = {
            self.get_data_var_name(key, variable) for variable in self.variables[key]
        }
        if set(dataset.data_vars) != expected_data_vars:
            raise ValueError(
                f"Persisted {key!r} variables are {sorted(dataset.data_vars)}, "
                f"expected {sorted(expected_data_vars)}."
            )

        required_coordinates = {"time", *field_space.index_names}
        missing_coordinates = required_coordinates - set(dataset.coords)
        if missing_coordinates:
            raise ValueError(
                f"Persisted {key!r} dataset is missing coordinates {sorted(missing_coordinates)}."
            )

        try:
            times = np.asarray(dataset.time.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Persisted {key!r} times must be numeric.") from exc
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError(f"Persisted {key!r} times must be a finite one-dimensional axis.")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError(f"Persisted {key!r} times must be strictly increasing.")

        for data_var in expected_data_vars:
            if tuple(dataset[data_var].dims) != ("time", "i"):
                raise ValueError(
                    f"Persisted {data_var!r} dimensions must be ('time', 'i'), "
                    f"got {dataset[data_var].dims}."
                )

        coefficient_multiindex = pd.MultiIndex.from_arrays(
            [dataset[name].values for name in field_space.index_names],
            names=field_space.index_names,
        )
        expected_multiindex = self._coefficient_indexes[key]
        if not coefficient_multiindex.equals(expected_multiindex):
            raise ValueError(
                f"Persisted {key!r} coefficient index does not match the simulation schema."
            )
        return coefficient_multiindex

    def load(self, key, store):
        """Load a persisted time-series dataset.

        Parameters
        ----------
        key : str
            The key identifying which time-series to load.
        """
        storage_kind = store.get_dataset_storage_kind(key)
        dataset = store.load_dataset(key)

        if dataset is not None:
            storage_representation = self.get_field_space(key).representation
            coefficient_multiindex = self._validate_loaded_dataset(key, dataset)
            coords = xr.Coordinates.from_pandas_multiindex(coefficient_multiindex, dim="i")
            self.datasets[key] = dataset.drop_vars(
                storage_representation.index_names
            ).assign_coords(coords)
            self._pending_start[key] = int(self.datasets[key].sizes.get("time", 0))
            self._full_save_required[key] = False
            if storage_kind is not None:
                self._storage_kinds[key] = storage_kind

    def add_entry(self, key, data, time):
        """Add entry to the time-series.

        Creates a new time-series if one does not exist, otherwise
        concatenates the new data along the time dimension.

        Parameters
        ----------
        key : str
            The key identifying the type of data.
        data : dict
            Dictionary of variables to set.
        time : float
            The time point for the data.
        """
        time_value = self._time_value(time)
        dataset = self._entry_dataset(key, data, time_value)
        self._merge_entry_dataset(key, dataset, time_value)

    def _entry_dataset(self, key, data, time_value):
        """Build one validated, time-indexed coefficient dataset."""
        expected_variables = set(self.variables[key])
        actual_variables = set(data)
        if actual_variables != expected_variables:
            raise ValueError(
                f"{key} entry has variables {sorted(actual_variables)}, "
                f"expected {sorted(expected_variables)}."
            )

        data_vars = {}
        for var in data:
            values = FieldCoefficients(self.field_spaces[key], data[var], name=f"{key}.{var}")
            data_vars[self.get_data_var_name(key, var)] = (
                ["time", "i"],
                values.to_vector().reshape((1, -1)),
            )

        return xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates.from_pandas_multiindex(
                self._coefficient_indexes[key], dim="i"
            ).merge({"time": [time_value]}),
        )

    def _merge_entry_dataset(self, key, dataset, time_value):
        """Insert or replace one entry while preserving sorted times."""
        existing = self.datasets.get(key)
        if existing is None or existing.sizes.get("time", 0) == 0:
            self.datasets[key] = dataset.sortby("time")
            self._pending_start[key] = 0
            self._full_save_required[key] = False
            return

        time_coords = np.asarray(existing.time.values, dtype=float)
        if time_value > float(time_coords[-1]) + TIME_TOLERANCE_SECONDS:
            previous_size = int(existing.sizes.get("time", 0))
            self.datasets[key] = xr.concat([existing, dataset], dim="time")
            pending_start = self._pending_start.get(key, previous_size)
            self._pending_start[key] = min(pending_start, previous_size)
            self._full_save_required[key] = bool(self._full_save_required.get(key, False))
            return

        replace = np.isclose(time_coords, time_value, rtol=0.0, atol=TIME_TOLERANCE_SECONDS)
        retained = existing.isel(time=np.flatnonzero(~replace))
        self.datasets[key] = xr.concat([retained, dataset], dim="time").sortby("time")
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
            previous_keys = [(key, var) for var in self.variables[key]]
            has_previous = all(item in self._previous_entries for item in previous_keys)
            changed = not has_previous or not all(
                np.allclose(
                    current_data[var],
                    self._previous_entries[(key, var)],
                    rtol=_VALUE_CHANGE_RTOL,
                    atol=0.0,
                )
                for var in self.variables[key]
            )
            if changed:
                for var in self.variables[key]:
                    self._previous_entries[(key, var)] = current_data[var]
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
        time = self._time_value(time)
        if np.any(self.datasets[key].time.values <= time + TIME_TOLERANCE_SECONDS):
            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.datasets[key].sel(
                time=[time + TIME_TOLERANCE_SECONDS], method="ffill"
            )
            dataset_before_time = float(dataset_before.time.item())

            for var in self.variables[key]:
                current_data[var] = dataset_before[
                    self.get_data_var_name(key, var)
                ].values.reshape(-1)

            # If requested, add linear interpolation correction.
            if (
                interpolation
                and dataset_before_time < time - TIME_TOLERANCE_SECONDS
                and np.any(self.datasets[key].time.values > time + TIME_TOLERANCE_SECONDS)
            ):
                dataset_after = self.datasets[key].sel(
                    time=[time + TIME_TOLERANCE_SECONDS], method="bfill"
                )
                for var in self.variables[key]:
                    current_data[var] += (
                        (time - dataset_before_time)
                        / (dataset_after.time.item() - dataset_before_time)
                        * (
                            dataset_after[self.get_data_var_name(key, var)].values.reshape(-1)
                            - dataset_before[self.get_data_var_name(key, var)].values.reshape(-1)
                        )
                    )

            return current_data
        else:
            # No data available for the specified time.
            return None

    def save(self, key, store, *, print_info: bool = False):
        """Persist one stored series to disk.

        Parameters
        ----------
        key : str
            The key identifying which time-series to save.
        """
        dataset = self.datasets[key].reset_index("i")
        time_size = int(dataset.sizes.get("time", 0))
        pending_start = int(self._pending_start.get(key, 0))
        full_save_required = bool(self._full_save_required.get(key, False))
        existing_storage_kind = store.get_dataset_storage_kind(key)
        target_storage_kind = self._storage_kinds.get(key)

        if target_storage_kind is None:
            target_storage_kind = (
                existing_storage_kind
                if existing_storage_kind is not None
                else store.default_dataset_storage_kind()
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
            store.save_dataset(
                dataset_to_save, key, print_info=print_info, storage="zarr", append_dim="time"
            )
        else:
            store.save_dataset(dataset, key, print_info=print_info, storage=target_storage_kind)

        self._pending_start[key] = time_size
        self._full_save_required[key] = False
        actual_storage_kind = store.get_dataset_storage_kind(key)
        self._storage_kinds[key] = (
            target_storage_kind if actual_storage_kind is None else actual_storage_kind
        )
