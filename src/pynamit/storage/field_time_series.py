"""Time-series storage for field coefficients."""

from collections.abc import Mapping
from heapq import merge
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from kompe.coefficients import CoefficientSpace
from kompe.math import get_array_module, to_numpy


def time_roundoff(time):
    """Return four float64 rounding units, not a physical time window.

    Relative simulation clocks can reach the same instant by different
    arithmetic, such as 0.3 and 3 * 0.1. Matching these representations
    must not merge distinct microsecond samples or anticipate input
    changes by a fixed physical duration.
    """
    return 4 * np.spacing(np.abs(np.asarray(time, dtype=float)))


class FieldTimeSeries:
    """Persist and select time-indexed field coefficients.

    Appends grow coefficient arrays geometrically. ``datasets`` exposes
    editable xarray views of occupied rows, not spare capacity.
    Use ``dataset.copy(deep=True)`` for an independent snapshot.
    """

    def __init__(self, field_spaces, variables, *, variable_attrs=None, time_origin=None):
        """Initialize named coefficient series from their field spaces.

        Parameters
        ----------
        field_spaces : dict
            Mapping from time-series group to ``CoefficientSpace``.
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
        self._append_buffers = {}
        self._pending_start: dict[str, int] = {}
        self._full_save_required: dict[str, bool] = {}
        self._saved_paths: dict[str, Path | None] = {}

        self._coefficient_layouts = self._build_coefficient_layouts()
        self._coefficient_coordinates = {
            key: self._build_coefficient_coordinates(key) for key in self.variables
        }

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
                    "field types belong in CoefficientSpace."
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
            if isinstance(group_spaces, CoefficientSpace):
                normalized[key] = group_spaces
                continue
            if not isinstance(group_spaces, Mapping):
                raise TypeError(
                    "FieldTimeSeries field_spaces values must be CoefficientSpace instances "
                    "or variable-to-CoefficientSpace mappings."
                )
            expected = set(self.variables[key])
            if set(group_spaces) != expected:
                raise ValueError(
                    f"Field spaces for {key!r} must use variables {sorted(expected)}."
                )
            if not all(isinstance(space, CoefficientSpace) for space in group_spaces.values()):
                raise TypeError(
                    "Variable field-space mappings must contain only CoefficientSpace instances."
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
                if isinstance(group_spaces, CoefficientSpace)
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
                    [
                        np.tile(values, field_space.component_count)
                        for values in field_space.index_arrays
                    ],
                    names=index_names,
                )
                layouts[key][variable] = {
                    "dimension": dimension,
                    "index_names": index_names,
                    "index": index,
                    "component_name": (
                        None
                        if field_space.representation == "scalar"
                        else "component"
                        if shared_layout
                        else f"{label}_component"
                    ),
                    "component_values": (
                        None
                        if field_space.representation == "scalar"
                        else np.repeat(
                            np.array([0, 1], dtype=np.int8), field_space.coefficient_count
                        )
                    ),
                }
        return layouts

    def _build_coefficient_coordinates(self, key):
        """Construct coefficient indexes once, independently of time."""
        coords = xr.Coordinates()
        indexes = {}
        for layout in self._coefficient_layouts[key].values():
            indexes.setdefault(layout["dimension"], layout["index"])
        for dimension, index in indexes.items():
            coords = coords.merge(xr.Coordinates.from_pandas_multiindex(index, dim=dimension))
        components = {}
        for layout in self._coefficient_layouts[key].values():
            if layout["component_name"] is not None:
                components.setdefault(
                    layout["component_name"], (layout["dimension"], layout["component_values"])
                )
        return coords.assign(components)

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
            attrs.setdefault(
                "field_type", "scalar" if field_space.representation == "scalar" else "tangential"
            )

        for layout in self._coefficient_layouts[key].values():
            component_name = layout["component_name"]
            if component_name is not None and component_name in dataset.coords:
                component_attrs = dataset.coords[component_name].attrs
                component_attrs.setdefault(
                    "long_name", "Helmholtz potential coefficient component"
                )
                component_attrs.setdefault("flag_values", [0, 1])
                component_attrs.setdefault(
                    "flag_meanings", "curl_free_potential divergence_free_potential"
                )
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
        if isinstance(group_spaces, CoefficientSpace):
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
        for key in self.variables:
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
            self._append_buffers.pop(key, None)
            self._pending_start[key] = int(self.datasets[key].sizes.get("time", 0))
            self._full_save_required[key] = False
            self._saved_paths[key] = store.existing_artifact_path(key)

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
        rows = {
            var: get_array_module(value).asarray(value)[None, ...] for var, value in data.items()
        }
        self.add_entries(key, rows, [time_value])

    def add_entries(self, key, data, times):
        """Add coefficient rows with a leading time axis.

        Build metadata and transfer each variable to the CPU once per
        batch. Unordered times are sorted; roundoff-equal times replace
        earlier entries in the supplied order, just as with add_entry.
        """
        times = np.asarray(times)
        if times.ndim != 1 or times.size == 0 or times.dtype.kind == "b":
            raise ValueError("times must be a non-empty 1-D numeric array.")
        times = times.astype(float)
        if not np.all(np.isfinite(times)):
            raise ValueError("times must be finite.")
        data_vars = self._coefficient_rows(key, data, times)
        existing = self.datasets.get(key)
        if (
            existing is not None
            and existing.sizes.get("time", 0)
            and times[0] > existing.time.values[-1] + time_roundoff(existing.time.values[-1])
            and np.all(np.diff(times) > time_roundoff(times[1:]))
        ):
            previous_size = existing.sizes["time"]
            self.datasets[key] = self._append_rows(key, existing, data_vars, times)
            self._pending_start[key] = min(
                self._pending_start.get(key, previous_size), previous_size
            )
            return

        coords = self._coefficient_coordinates[key].assign(time=times)
        dataset = self._apply_metadata(key, xr.Dataset(data_vars=data_vars, coords=coords))
        ordered_times = np.sort(times)
        if np.any(np.diff(ordered_times) <= time_roundoff(ordered_times[1:])):
            # Tolerant equality is not transitive. Preserve insertion
            # order for overlapping replacements within this batch.
            for index in range(times.size):
                self._merge_entries(key, dataset.isel(time=slice(index, index + 1)))
        else:
            self._merge_entries(key, dataset.isel(time=np.argsort(times)))

    def _coefficient_rows(self, key, data, times):
        """Validate rows and transfer them to owned CPU arrays."""
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
            xp = get_array_module(data[var])
            rows = xp.asarray(data[var])
            if rows.ndim < 2 or rows.shape[0] != times.size:
                raise ValueError(f"{key}.{var} must have {times.size} coefficient rows.")
            # Numerical field axes lead; persisted time rows lead.
            values = rows.reshape(times.size, field_space.size).T
            values = field_space.project_mean_free(values, name=f"{key}.{var}")
            values = values.reshape(field_space.size, times.size).T
            dimension = self._coefficient_layouts[key][var]["dimension"]
            # Own the CPU rows even when normalization is an identity.
            data_vars[self.get_data_var_name(key, var)] = (
                ["time", dimension],
                np.array(to_numpy(values), copy=True),
            )

        return data_vars

    def _merge_entries(self, key, dataset):
        """Merge a sorted batch with no near-equal internal times."""
        existing = self.datasets.get(key)
        if existing is None or existing.sizes.get("time", 0) == 0:
            self._append_buffers.pop(key, None)
            self.datasets[key] = dataset
            self._pending_start[key] = 0
            self._full_save_required[key] = False
            return

        time_coords = np.asarray(existing.time.values, dtype=float)
        new_times = dataset.time.values
        positions = np.searchsorted(new_times, time_coords)
        before = new_times[np.maximum(positions - 1, 0)]
        after = new_times[np.minimum(positions, new_times.size - 1)]
        replace = np.abs(time_coords - before) <= time_roundoff(
            np.maximum(np.abs(time_coords), np.abs(before))
        )
        replace |= np.abs(time_coords - after) <= time_roundoff(
            np.maximum(np.abs(time_coords), np.abs(after))
        )
        retained = existing.isel(time=np.flatnonzero(~replace))
        combined = xr.concat([retained, dataset], dim="time", coords="minimal", join="exact")
        self.datasets[key] = combined.isel(time=np.argsort(combined.time.values))
        self._append_buffers.pop(key, None)
        self._pending_start[key] = 0
        self._full_save_required[key] = True

    def _append_rows(self, key, existing, rows, new_times):
        """Append rows without copying the history on every write.

        Double capacity when reallocating; otherwise copy only new rows.
        The exposed array identifies each live view. Replaced xarray
        variables are copied back before appending, so interactive edits
        remain authoritative rather than being hidden by a stale buffer.
        """
        for dimension, layout in {
            layout["dimension"]: layout for layout in self._coefficient_layouts[key].values()
        }.items():
            if not existing.get_index(dimension).equals(layout["index"]):
                raise ValueError("Coefficient indexes must match the field space when appending.")
        buffers = self._append_buffers.setdefault(key, {})
        previous_size = existing.sizes["time"]
        size = previous_size + new_times.size
        variables = {}
        for name in existing.data_vars:
            variable = existing.variables[name]
            previous = variable.values
            _, values = rows[name]
            dtype = np.result_type(previous.dtype, values.dtype)
            buffer, exposed = buffers.get(name, (None, None))
            if buffer is None or buffer.shape[0] < size or buffer.dtype != dtype:
                capacity = max(size, 2 * previous_size)
                buffer = np.empty((capacity,) + previous.shape[1:], dtype=dtype)
                np.copyto(buffer[:previous_size], previous)
            elif previous is not exposed:
                np.copyto(buffer[:previous_size], previous)
            buffer[previous_size:size] = values
            view = buffer[:size]
            variables[name] = xr.Variable(
                variable.dims, view, attrs=variable.attrs, encoding=variable.encoding
            )
            buffers[name] = (buffer, view)
        # Keep ordinary xarray time indexes; reserve spare capacity only
        # for the much larger coefficient arrays.
        times = np.concatenate([existing.time.values, new_times])
        time = xr.Variable(
            ("time",), times, attrs=existing.time.attrs, encoding=existing.time.encoding
        )
        coords = existing.coords.drop_vars("time").assign(time=time)
        return xr.Dataset(variables, coords=coords, attrs=existing.attrs)

    def get_entry(self, key, time, interpolation=False, *, variables=None, fill_value=None):
        """Select coefficients at one time or a non-empty time array.

        Owned arrays have ``CoefficientSpace.shape`` followed by a time
        axis for 1-D queries. Query order and repeats are retained.
        Only requested rows are loaded from lazy storage.

        Held values are right-continuous, matching float64 timestamp
        roundoff. Interpolation is optional; the final value is held
        after the final sample. Neither mode borrows a future value:
        return None if any requested time precedes the first sample,
        or fill those columns with an explicit ``fill_value``.
        Unknown stream/variable names raise KeyError.

        ``variables`` selects variables (None means all). An empty
        sequence checks availability without loading coefficients.
        ``fill_value=0`` suits a source defined as zero before onset;
        this is not an implicit missing-data policy.
        """
        batch = np.ndim(time) != 0
        if batch:
            time = np.asarray(time)
            if time.ndim != 1 or not time.size or time.dtype.kind not in "fiu":
                raise ValueError("time must be a scalar or non-empty numeric 1-D array.")
            time = time.astype(float, copy=False)
            if not np.all(np.isfinite(time)):
                raise ValueError("time must contain only finite values.")
        else:
            time = self._time_value(time)
        known_variables = self.variables[key]
        variables = known_variables if variables is None else tuple(variables)
        unknown = set(variables) - set(known_variables)
        if unknown:
            raise KeyError(f"Unknown variables in {key!r}: {sorted(unknown)}.")
        batch_shape = (time.size,) if batch else ()
        dataset = self.datasets.get(key)
        if dataset is None:
            return (
                None
                if fill_value is None
                else {
                    var: np.full(self.get_field_space(key, var).shape + batch_shape, fill_value)
                    for var in variables
                }
            )

        times = dataset.time.values
        before = np.searchsorted(times, time + time_roundoff(time), side="right") - 1
        missing = before < 0
        has_missing = np.any(missing) if batch else missing
        if has_missing and fill_value is None:
            return None
        if has_missing and (np.all(missing) if batch else True):
            return {
                var: np.full(self.get_field_space(key, var).shape + batch_shape, fill_value)
                for var in variables
            }

        # Share brackets across variables and read unique rows only.
        # Scalar sampling stays a single slice, also during evolution.
        if batch:
            before = np.maximum(before, 0)
            after = np.minimum(before + 1, times.size - 1)
            interpolate = (
                interpolation
                & (after > before)
                & (times[before] < time - time_roundoff(time))
                & ~missing
            )
            following = np.where(interpolate, after, before)
            selected, inverse = np.unique(np.concatenate([before, following]), return_inverse=True)
            previous_rows, next_rows = inverse.reshape(2, -1)
            has_interpolation = np.any(interpolate)
            fraction = np.zeros(time.size)
            fraction[interpolate] = (time[interpolate] - times[before[interpolate]]) / (
                times[after[interpolate]] - times[before[interpolate]]
            )
        else:
            before_time = float(times[before])
            interpolate = (
                interpolation
                and before + 1 < times.size
                and before_time < time - time_roundoff(time)
            )
            selected = slice(before, before + (2 if interpolate else 1))
            if interpolate:
                fraction = (time - before_time) / (float(times[before + 1]) - before_time)

        current_data = {}
        for var in variables:
            values = dataset.variables[self.get_data_var_name(key, var)][selected].values
            shape = self.get_field_space(key, var).shape
            if batch:
                current = values[previous_rows]
                if has_interpolation:
                    scale = fraction.astype(np.result_type(current.dtype, 0.0), copy=False)
                    current = current + scale[:, None] * (values[next_rows] - current)
                if has_missing:
                    current = np.where(missing[:, None], fill_value, current)
                current_data[var] = current.T.reshape(shape + batch_shape).copy()
            else:
                previous = values[0].reshape(shape)
                current_data[var] = (
                    previous + fraction * (values[1].reshape(shape) - previous)
                    if interpolate
                    else previous.copy()
                )
        return current_data

    def iter_intervals(self, start, stop, *, previous=None):
        """Yield ``(left, right, entries)`` for constant held inputs.

        Entries map every known stream to owned coefficient arrays,
        or None before its first record. Adjacent rows are read once,
        lazily; identical records do not split an interval. Unchanged
        entries retain object identity, including against an optional
        previous selection from this same series. Treat yielded arrays
        as read-only. A new iterator sees edits to the live datasets.
        Do not edit streams while consuming an iterator.

        Intervals are left-closed and right-open. A final zero-length
        interval at ``stop`` supplies inputs for endpoint diagnostics,
        including changes exactly at stop. Times differing only by
        float64 roundoff share a boundary, as in get_entry.
        """
        start, stop = self._time_value(start), self._time_value(stop)
        if abs(stop - start) <= time_roundoff(start):
            stop = start
        if stop < start:
            raise ValueError("stop must be at or after start.")
        previous = {} if previous is None else previous

        def changes(key):
            dataset = self.datasets.get(key)
            if dataset is None:
                yield start, key, None
                return
            times = dataset.time.values
            first = int(np.searchsorted(times, start + time_roundoff(start), side="right")) - 1
            last = int(np.searchsorted(times, stop + time_roundoff(stop), side="right"))
            old = previous.get(key)
            if first < 0:
                yield start, key, None
                old = None
            for index in range(max(first, 0), last):
                values = {
                    name: dataset.variables[self.get_data_var_name(key, name)][
                        index
                    ].values.reshape(self.get_field_space(key, name).shape)
                    for name in self.variables[key]
                }
                same = old is not None and all(
                    np.array_equal(value, old[name], equal_nan=True)
                    for name, value in values.items()
                )
                if not same:
                    old = {name: value.copy() for name, value in values.items()}
                if index == first or not same:
                    time = start if index == first else float(times[index])
                    if abs(time - stop) <= time_roundoff(stop):
                        time = stop
                    yield time, key, old

        events = merge(*(changes(key) for key in self.variables), key=lambda event: event[0])
        left, entries = start, {}
        for time, key, values in events:
            if time > left + time_roundoff(left):
                yield left, time, entries
                entries = entries.copy()
                left = time
            entries[key] = values
        yield left, stop, entries
        if left < stop:
            yield stop, stop, entries

    def save(self, key, store, *, incremental: bool = True, print_info: bool = False):
        """Persist one complete series to the destination store.

        Append and no-change optimizations apply only to the last
        saved or loaded artifact. A different destination receives
        the full series, preserving its existing format or using the
        destination store's preferred format for a new artifact.
        Set ``incremental=False`` to include direct edits to live
        datasets that did not pass through ``add_entry``.

        Parameters
        ----------
        key : str
            The key identifying which time-series to save.
        store : ArtifactStore
            Destination for the named time-series artifact.
        """
        dataset = self.datasets[key]
        time_size = int(dataset.sizes.get("time", 0))
        pending_start = int(self._pending_start.get(key, 0))
        full_save_required = not incremental or bool(self._full_save_required.get(key, False))
        existing_storage_kind = store.get_dataset_storage_kind(key)
        target_storage_kind = (
            existing_storage_kind
            if existing_storage_kind is not None
            else store.default_dataset_storage_kind()
        )
        same_artifact = existing_storage_kind is not None and store.existing_artifact_path(
            key
        ) == self._saved_paths.get(key)

        if same_artifact and not full_save_required and pending_start >= time_size:
            return

        index_dimensions = sorted(
            {layout["dimension"] for layout in self._coefficient_layouts[key].values()}
        )
        dataset = dataset.reset_index(index_dimensions)
        if (
            same_artifact
            and target_storage_kind == "zarr"
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
        self._saved_paths[key] = store.existing_artifact_path(key)
