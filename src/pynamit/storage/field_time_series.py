"""Time-series storage for field coefficients."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr
from kompe.math import to_numpy

from pynamit.fields import FieldCoefficients, FieldSpace

TIME_TOLERANCE_SECONDS = 1e-6
_VALUE_CHANGE_RTOL = 1e-6


class FieldTimeSeries:
    """Persist and select time-indexed field coefficients."""

    def __init__(self, field_spaces, variables, *, variable_attrs=None, time_origin=None):
        """Initialize named coefficient series from their field spaces.

        Parameters
        ----------
        field_spaces : dict
            Mapping from time-series group to ``FieldSpace``.
        variables : dict
            Variable names for each group.
        variable_attrs : dict, optional
            Physical xarray attributes for each group and variable.
        time_origin : str, optional
            UTC origin from which simulation times are measured.
        """
        self.variables = self._normalize_variables(variables)
        self.field_spaces = self._normalize_field_spaces(field_spaces)
        self._variable_field_spaces = self._expand_variable_field_spaces()
        self.variable_attrs = self._normalize_variable_attrs(variable_attrs)
        self.time_origin = None if time_origin is None else str(time_origin)

        # Initialize in-memory series and persistence bookkeeping.
        self.datasets = {}
        self._previous_entries = {}
        self._pending_start: dict[str, int] = {}
        self._full_save_required: dict[str, bool] = {}
        self._storage_kinds: dict[str, str] = {}

        self._coefficient_layouts = self._build_coefficient_layouts()

    def _normalize_variable_attrs(self, variable_attrs):
        """Return complete copied variable-attribute mappings."""
        if variable_attrs is None:
            return {key: {name: {} for name in names} for key, names in self.variables.items()}
        if set(variable_attrs) != set(self.variables):
            raise ValueError("Variable attributes and variables must use the same group keys.")
        normalized = {}
        for key, names in self.variables.items():
            group_attrs = variable_attrs[key]
            if set(group_attrs) != set(names):
                raise ValueError(
                    f"Variable attributes for {key!r} must use variables {sorted(names)}."
                )
            normalized[key] = {name: dict(group_attrs[name]) for name in names}
        return normalized

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
        for key, group_spaces in field_spaces.items():
            if isinstance(group_spaces, FieldSpace):
                normalized[key] = group_spaces
                continue
            if not isinstance(group_spaces, Mapping):
                raise TypeError(
                    "FieldTimeSeries field_spaces values must be FieldSpace instances "
                    "or variable-to-FieldSpace mappings."
                )
            expected = set(self.variables[key])
            if set(group_spaces) != expected:
                raise ValueError(
                    f"Field spaces for {key!r} must use variables {sorted(expected)}."
                )
            if not all(isinstance(space, FieldSpace) for space in group_spaces.values()):
                raise TypeError(
                    "Variable field-space mappings must contain only FieldSpace instances."
                )
            normalized[key] = dict(group_spaces)
        return normalized

    def _expand_variable_field_spaces(self):
        """Return one explicit field space for every stored variable."""
        expanded = {}
        for key, names in self.variables.items():
            group_spaces = self.field_spaces[key]
            expanded[key] = (
                {name: group_spaces for name in names}
                if isinstance(group_spaces, FieldSpace)
                else dict(group_spaces)
            )
        return expanded

    def _build_coefficient_layouts(self):
        """Build coefficient dimensions and indexes by variable."""
        layouts = {}
        for key, variable_spaces in self._variable_field_spaces.items():
            signatures = {space.signature for space in variable_spaces.values()}
            shared_layout = len(signatures) == 1
            kinds = [space.kind.lower() for space in variable_spaces.values()]
            unique_kinds = len(set(kinds)) == len(signatures)
            signature_labels = {}
            for variable, field_space in variable_spaces.items():
                signature = field_space.signature
                if signature in signature_labels:
                    continue
                signature_labels[signature] = (
                    field_space.kind.lower() if unique_kinds else variable.lower()
                )

            layouts[key] = {}
            for variable, field_space in variable_spaces.items():
                label = signature_labels[field_space.signature]
                dimension = "i" if shared_layout else f"{label}_i"
                index_names = tuple(
                    field_space.index_names
                    if shared_layout
                    else (f"{label}_{name}" for name in field_space.index_names)
                )
                index = pd.MultiIndex.from_arrays(
                    field_space.multiindex_arrays(), names=index_names
                )
                layouts[key][variable] = {
                    "dimension": dimension,
                    "index_names": index_names,
                    "index": index,
                    "component_name": (
                        None
                        if field_space.field_type == "scalar"
                        else "component"
                        if shared_layout
                        else f"{label}_component"
                    ),
                    "component_values": (
                        None
                        if field_space.field_type == "scalar"
                        else np.repeat(np.array([0, 1], dtype=np.int8), field_space.index_length)
                    ),
                }
        return layouts

    def _apply_metadata(self, key, dataset):
        """Attach physical metadata while preserving stored values."""
        dataset.coords["time"].attrs.setdefault("units", "s")
        dataset.coords["time"].attrs.setdefault("long_name", "simulation time since t0")
        if self.time_origin is not None:
            dataset.coords["time"].attrs.setdefault("time_origin", self.time_origin)

        for variable in self.variables[key]:
            data_var = self.get_data_var_name(key, variable)
            field_space = self.get_field_space(key, variable)
            attrs = dataset[data_var].attrs
            for name, value in self.variable_attrs[key][variable].items():
                attrs.setdefault(name, value)
            attrs.setdefault("physical_name", variable)
            attrs.setdefault("coefficient_basis", field_space.kind)
            attrs.setdefault("field_type", field_space.field_type)

        for layout in self._coefficient_layouts[key].values():
            component_name = layout["component_name"]
            if component_name is not None and component_name in dataset.coords:
                component_attrs = dataset.coords[component_name].attrs
                component_attrs.setdefault("long_name", "Helmholtz coefficient component")
                component_attrs.setdefault("flag_values", [0, 1])
                component_attrs.setdefault("flag_meanings", "curl_free divergence_free")
            for coordinate_name in layout["index_names"]:
                if coordinate_name not in dataset.coords:
                    continue
                base_name = coordinate_name.rsplit("_", 1)[-1]
                if base_name in {"n", "m"}:
                    dataset.coords[coordinate_name].attrs.setdefault("units", "1")
                elif base_name in {"theta", "phi"}:
                    dataset.coords[coordinate_name].attrs.setdefault("units", "degrees")
        return dataset

    def get_field_space(self, key, variable=None):
        """Return a group or variable field space."""
        group_spaces = self.field_spaces[key]
        if isinstance(group_spaces, FieldSpace):
            return group_spaces
        if variable is not None:
            return group_spaces[variable]
        spaces = tuple(group_spaces.values())
        if spaces and all(space.signature == spaces[0].signature for space in spaces[1:]):
            return spaces[0]
        raise ValueError(f"{key!r} contains multiple field spaces; specify a variable name.")

    def get_data_var_name(self, key, var):
        """Return the xarray variable name for one series variable."""
        return f"{self.get_field_space(key, var).kind}_{var}"

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
        expected_data_vars = {
            self.get_data_var_name(key, variable) for variable in self.variables[key]
        }
        if set(dataset.data_vars) != expected_data_vars:
            raise ValueError(
                f"Persisted {key!r} variables are {sorted(dataset.data_vars)}, "
                f"expected {sorted(expected_data_vars)}."
            )

        required_coordinates = {"time"}
        for layout in self._coefficient_layouts[key].values():
            required_coordinates.update(layout["index_names"])
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

        for variable in self.variables[key]:
            data_var = self.get_data_var_name(key, variable)
            dimension = self._coefficient_layouts[key][variable]["dimension"]
            if tuple(dataset[data_var].dims) != ("time", dimension):
                raise ValueError(
                    f"Persisted {data_var!r} dimensions must be ('time', {dimension!r}), "
                    f"got {dataset[data_var].dims}."
                )

            layout = self._coefficient_layouts[key][variable]
            component_name = layout["component_name"]
            if component_name is not None and component_name in dataset.coords:
                component = dataset.coords[component_name]
                if component.dims != (dimension,) or not np.array_equal(
                    component.values, layout["component_values"]
                ):
                    raise ValueError(
                        f"Persisted {component_name!r} labels do not match the "
                        "tangential coefficient layout."
                    )

        indexes = {}
        for layout in self._coefficient_layouts[key].values():
            dimension = layout["dimension"]
            if dimension in indexes:
                continue
            index = pd.MultiIndex.from_arrays(
                [dataset[name].values for name in layout["index_names"]],
                names=layout["index_names"],
            )
            if not index.equals(layout["index"]):
                raise ValueError(
                    f"Persisted {key!r} coefficient index does not match the simulation schema."
                )
            indexes[dimension] = index
        return indexes

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
            coefficient_indexes = self._validate_loaded_dataset(key, dataset)
            index_names = {
                name
                for layout in self._coefficient_layouts[key].values()
                for name in layout["index_names"]
            }
            restored = dataset.drop_vars(index_names)
            for dimension, index in coefficient_indexes.items():
                restored = restored.assign_coords(
                    xr.Coordinates.from_pandas_multiindex(index, dim=dimension)
                )
            missing_components = {
                layout["component_name"]: (layout["dimension"], layout["component_values"])
                for layout in self._coefficient_layouts[key].values()
                if layout["component_name"] is not None
                and layout["component_name"] not in restored.coords
            }
            restored = restored.assign_coords(missing_components)
            self.datasets[key] = self._apply_metadata(key, restored)
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
            field_space = self.get_field_space(key, var)
            values = FieldCoefficients(field_space, data[var], name=f"{key}.{var}")
            dimension = self._coefficient_layouts[key][var]["dimension"]
            data_vars[self.get_data_var_name(key, var)] = (
                ["time", dimension],
                to_numpy(values.to_vector()).reshape((1, -1)),
            )

        coords = xr.Coordinates({"time": [time_value]})
        indexes = {}
        for layout in self._coefficient_layouts[key].values():
            indexes.setdefault(layout["dimension"], layout["index"])
        for dimension, index in indexes.items():
            coords = coords.merge(xr.Coordinates.from_pandas_multiindex(index, dim=dimension))
        component_coordinates = {}
        for layout in self._coefficient_layouts[key].values():
            component_name = layout["component_name"]
            if component_name is not None:
                component_coordinates.setdefault(
                    component_name, (layout["dimension"], layout["component_values"])
                )
        coords = coords.assign(component_coordinates)
        return self._apply_metadata(key, xr.Dataset(data_vars=data_vars, coords=coords))

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
        index_dimensions = sorted(
            {layout["dimension"] for layout in self._coefficient_layouts[key].values()}
        )
        dataset = self.datasets[key].reset_index(index_dimensions)
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
