"""Read and remap ReMIX field-aligned current for MAGE forcing."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import h5py
import numpy as np

from pynamit.coordinates import wrap_longitude_180
from pynamit.workflows.mage.gamera import _datetime_from_mjd, _geographic_grid_in_sm, _h5_text

REMIX_TIME_TOLERANCE_SECONDS = 1.0e-3


class _RemixGridInterpolator:
    """Interpolate one saved ReMIX hemisphere on its native tensor grid.

    Kaiju's ReMIX coupling uses a four-point interpolant in colatitude
    and longitude, with a three-vertex rule in the cell touching the
    pole. ReMIX writes fields without that degenerate pole and stores a
    staggered X/Y grid whose cell centers locate the remaining field
    nodes. This class reconstructs the pole and applies the same mapping
    geometry.
    """

    def __init__(self, source_lat: np.ndarray, source_lon: np.ndarray) -> None:
        source_lat, source_lon = np.broadcast_arrays(
            np.asarray(source_lat, dtype=float), np.asarray(source_lon, dtype=float)
        )
        if source_lat.ndim != 2 or min(source_lat.shape) < 2:
            raise ValueError(
                "A ReMIX grid must be two-dimensional with at least two cells per axis."
            )
        if np.any(~np.isfinite(source_lat)) or np.any(~np.isfinite(source_lon)):
            raise ValueError("ReMIX grid coordinates must be finite.")

        latitude = source_lat[:, 0]
        longitude = np.mod(source_lon[0], 360.0)
        longitude_residual = wrap_longitude_180(source_lon - source_lon[[0]])
        if not np.allclose(source_lat, latitude[:, None], rtol=0.0, atol=1e-12) or not np.allclose(
            longitude_residual, 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "Saved ReMIX coordinates must form a rectilinear latitude/longitude grid."
            )
        if not (np.all(latitude > 0.0) or np.all(latitude < 0.0)):
            raise ValueError("A saved ReMIX grid must contain exactly one magnetic hemisphere.")

        self._source_shape = source_lat.shape
        self._latitude_order = np.argsort(latitude)
        self._longitude_order = np.argsort(longitude)
        self._latitude = latitude[self._latitude_order]
        self._longitude = longitude[self._longitude_order]
        if np.any(np.diff(self._latitude) <= 0.0) or np.any(np.diff(self._longitude) <= 0.0):
            raise ValueError("ReMIX latitude and longitude coordinates must be unique.")

    def interpolate(
        self, values: np.ndarray, target_lon: np.ndarray, target_lat: np.ndarray
    ) -> np.ndarray:
        """Interpolate periodically within the source hemisphere."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._source_shape:
            raise ValueError(
                f"ReMIX field shape {values.shape} does not match {self._source_shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("ReMIX interpolation requires finite source values.")
        values = values[np.ix_(self._latitude_order, self._longitude_order)]

        # ReMIX omits the degenerate pole when writing fields and
        # restores it as the mean of the poleward ring when reading.
        latitude = self._latitude
        if latitude[0] > 0.0:
            poleward_ring = values[-1]
            latitude = np.concatenate((latitude, [90.0]))
            pole_value = np.mean(poleward_ring)
            values = np.vstack((values, np.full((1, values.shape[1]), pole_value)))
            north = True
        else:
            poleward_ring = values[0]
            latitude = np.concatenate(([-90.0], latitude))
            pole_value = np.mean(poleward_ring)
            values = np.vstack((np.full((1, values.shape[1]), pole_value), values))
            north = False

        target_lon, target_lat = np.broadcast_arrays(
            np.asarray(target_lon, dtype=float), np.asarray(target_lat, dtype=float)
        )
        target_shape = target_lat.shape
        query_latitude = target_lat.reshape(-1)
        query_longitude = (
            np.mod(target_lon.reshape(-1) - self._longitude[0], 360.0) + self._longitude[0]
        )
        finite = np.isfinite(query_latitude) & np.isfinite(query_longitude)
        latitude_tolerance = max(
            1e-12, 16.0 * np.finfo(float).eps * float(np.max(np.abs(latitude)))
        )
        covered = (
            finite
            & (query_latitude >= latitude[0] - latitude_tolerance)
            & (query_latitude <= latitude[-1] + latitude_tolerance)
        )
        result = np.full(query_latitude.size, np.nan)
        if not np.any(covered):
            return result.reshape(target_shape)

        covered_indices = np.flatnonzero(covered)
        query_latitude = np.clip(query_latitude[covered], latitude[0], latitude[-1])
        query_latitude = np.where(
            np.abs(query_latitude - self._latitude[0]) <= latitude_tolerance,
            self._latitude[0],
            query_latitude,
        )
        query_latitude = np.where(
            np.abs(query_latitude - self._latitude[-1]) <= latitude_tolerance,
            self._latitude[-1],
            query_latitude,
        )
        query_longitude = query_longitude[covered]
        latitude_index = np.searchsorted(latitude, query_latitude, side="right") - 1
        latitude_index = np.clip(latitude_index, 0, latitude.size - 2)

        periodic_longitude = np.concatenate((self._longitude, [self._longitude[0] + 360.0]))
        longitude_index = np.searchsorted(periodic_longitude, query_longitude, side="right") - 1
        longitude_index = np.clip(longitude_index, 0, self._longitude.size - 1)
        next_longitude_index = (longitude_index + 1) % self._longitude.size

        latitude_fraction = (query_latitude - latitude[latitude_index]) / (
            latitude[latitude_index + 1] - latitude[latitude_index]
        )
        next_longitude = periodic_longitude[longitude_index + 1]
        longitude_fraction = (query_longitude - periodic_longitude[longitude_index]) / (
            next_longitude - periodic_longitude[longitude_index]
        )

        lower_left = values[latitude_index, longitude_index]
        lower_right = values[latitude_index, next_longitude_index]
        upper_left = values[latitude_index + 1, longitude_index]
        upper_right = values[latitude_index + 1, next_longitude_index]
        result[covered_indices] = (1.0 - latitude_fraction) * (
            (1.0 - longitude_fraction) * lower_left + longitude_fraction * lower_right
        ) + latitude_fraction * (
            (1.0 - longitude_fraction) * upper_left + longitude_fraction * upper_right
        )

        # Kaiju treats the polar quadrilateral as a triangle because all
        # longitude vertices at the pole are one physical point. Its map
        # therefore averages the reconstructed pole and the two adjacent
        # values on the poleward ring, independently of polar distance.
        polar_cap = (
            query_latitude > self._latitude[-1] if north else query_latitude < self._latitude[0]
        )
        if np.any(polar_cap):
            result[covered_indices[polar_cap]] = (
                pole_value
                + poleward_ring[longitude_index[polar_cap]]
                + poleward_ring[next_longitude_index[polar_cap]]
            ) / 3.0
        return result.reshape(target_shape)


def _combine_remix_hemispheres(south: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Merge hemispheres and set the uncovered low latitudes to zero."""
    output = np.array(south, copy=True)
    mask = np.isnan(output)
    output[mask] = north[mask]
    output[np.isnan(output)] = 0.0
    if np.any(~np.isfinite(output)):
        raise ValueError("REMIX FAC interpolation produced non-finite values.")
    return output


def _dipole_radial_direction_cosine(magnetic_latitude: np.ndarray) -> np.ndarray:
    """Return a centered dipole's absolute radial direction cosine."""
    sin_latitude = np.sin(np.deg2rad(np.asarray(magnetic_latitude, dtype=float)))
    return np.abs(2.0 * sin_latitude / np.sqrt(1.0 + 3.0 * sin_latitude**2))


def _upward_fac_to_radial_current(
    upward_fac: np.ndarray, magnetic_latitude: np.ndarray
) -> np.ndarray:
    """Convert upward-positive dipole FAC to outward current."""
    return np.asarray(upward_fac, dtype=float) * _dipole_radial_direction_cosine(magnetic_latitude)


def _remix_upward_fac_source(
    hemisphere: str,
    fac: np.ndarray,
    unsigned_magnetic_latitude: np.ndarray,
    grid_longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical SM positions and upward-positive ReMIX FAC.

    ReMIX stores both hemispheres on the same unsigned polar grid. Kaiju
    interprets southern positions with latitude ``-latitude`` and
    longitude ``-longitude``. Its saved FAC is parallel-positive, so it
    is negated in the north and retained in the south to obtain one
    outward/upward-positive convention.
    """
    hemisphere = str(hemisphere).upper()
    if hemisphere not in {"NORTH", "SOUTH"}:
        raise ValueError("ReMIX hemisphere must be 'NORTH' or 'SOUTH'.")

    latitude, longitude = np.broadcast_arrays(
        np.asarray(unsigned_magnetic_latitude, dtype=float),
        np.asarray(grid_longitude, dtype=float),
    )
    fac = np.asarray(fac, dtype=float)
    if fac.shape != latitude.shape:
        raise RuntimeError(
            f"ReMIX {hemisphere} FAC shape {fac.shape} does not match "
            f"the cell-center grid {latitude.shape}."
        )
    if np.any(~np.isfinite(fac)):
        raise RuntimeError(f"ReMIX {hemisphere} FAC must be finite.")

    if hemisphere == "NORTH":
        return latitude, wrap_longitude_180(longitude), -fac
    return -latitude, wrap_longitude_180(-longitude), fac


def _remix_cell_center_coordinates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return saved ReMIX field-node latitude and longitude."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 2 or min(x.shape) < 3 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise RuntimeError("ReMIX X/Y must be finite two-dimensional corner grids.")
    x_center = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    y_center = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])
    polar_radius = np.hypot(x_center, y_center)
    tolerance = 32.0 * np.finfo(float).eps
    if np.any(polar_radius > 1.0 + tolerance):
        raise RuntimeError("ReMIX X/Y cell centres must lie inside the unit polar disk.")
    colatitude = np.arcsin(np.clip(polar_radius, 0.0, 1.0))
    longitude = np.arctan2(y_center, x_center)
    return 90.0 - np.degrees(colatitude), wrap_longitude_180(np.degrees(longitude))


class _RemixRadialCurrentReader:
    """Read only the ReMIX FAC needed for outward current forcing."""

    def __init__(self, remix_file: Path) -> None:
        self._remix_file = Path(remix_file)
        self._file: h5py.File | None = None
        self._unsigned_latitude: np.ndarray | None = None
        self._grid_longitude: np.ndarray | None = None
        self._interpolators: dict[str, _RemixGridInterpolator] = {}

    def __enter__(self):
        """Open and validate the reusable ReMIX grid."""
        self._file = h5py.File(self._remix_file, "r")
        try:
            if _h5_text(self._file.attrs.get("UnitsID", "")) != "ReMIX":
                raise RuntimeError("ReMIX forcing must declare UnitsID='ReMIX'.")
            missing = [name for name in ("X", "Y") if name not in self._file]
            if missing:
                raise RuntimeError(f"ReMIX forcing is missing grid datasets {missing}.")
            wrong_units = [
                name
                for name in ("X", "Y")
                if _h5_text(self._file[name].attrs.get("Units", "")) != "Ri"
            ]
            if wrong_units:
                raise RuntimeError(f"ReMIX grid datasets must use Ri units: {wrong_units}.")
            unsigned_latitude, grid_longitude = _remix_cell_center_coordinates(
                self._file["X"][:], self._file["Y"][:]
            )
            self._unsigned_latitude = unsigned_latitude
            self._grid_longitude = grid_longitude
            shape = unsigned_latitude.shape
            zeros = np.zeros(shape)
            for hemisphere in ("NORTH", "SOUTH"):
                source_lat, source_lon, _ = _remix_upward_fac_source(
                    hemisphere, zeros, unsigned_latitude, grid_longitude
                )
                self._interpolators[hemisphere] = _RemixGridInterpolator(source_lat, source_lon)
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the ReMIX source file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _history(self, step: int) -> tuple[h5py.Group, dt.datetime]:
        """Return one ReMIX history and its exact source time."""
        if self._file is None:
            raise RuntimeError("ReMIX reader must be used as a context manager.")
        group_name = f"Step#{step}"
        if group_name not in self._file:
            raise RuntimeError(f"ReMIX forcing is missing {group_name!r}.")
        history = self._file[group_name]
        if "MJD" not in history.attrs:
            raise RuntimeError(f"ReMIX history {group_name!r} is missing MJD time metadata.")
        source_time = _datetime_from_mjd(history.attrs["MJD"])
        return history, source_time

    def source_time(self, step: int) -> dt.datetime:
        """Return the exact timestamp of one ReMIX history."""
        return self._history(step)[1]

    @property
    def equatorward_sm_latitude(self) -> float:
        """Return the saved ReMIX grid's equatorward SM latitude."""
        if self._unsigned_latitude is None:
            raise RuntimeError("ReMIX reader must be used as a context manager.")
        return float(np.min(np.abs(self._unsigned_latitude)))

    @staticmethod
    def _fac(history: h5py.Group, hemisphere: str) -> np.ndarray:
        """Read one parallel-positive ReMIX FAC field."""
        dataset_name = f"Field-aligned current {hemisphere}"
        if dataset_name not in history:
            raise RuntimeError(f"ReMIX history is missing {dataset_name!r}.")
        dataset = history[dataset_name]
        if _h5_text(dataset.attrs.get("Units", "")) != "muA/m**2":
            raise RuntimeError(f"ReMIX {dataset_name!r} must use muA/m**2 units.")
        return np.asarray(dataset, dtype=float)

    def _hemisphere(
        self,
        hemisphere: str,
        fac: np.ndarray,
        target_sm_lon: np.ndarray,
        target_sm_lat: np.ndarray,
    ) -> np.ndarray:
        """Sample one FAC hemisphere at target SM positions."""
        if self._unsigned_latitude is None or self._grid_longitude is None:
            raise RuntimeError("ReMIX reader must be used as a context manager.")
        _, _, upward_fac = _remix_upward_fac_source(
            hemisphere, fac, self._unsigned_latitude, self._grid_longitude
        )
        return self._interpolators[hemisphere].interpolate(
            upward_fac, target_sm_lon, target_sm_lat
        )

    def read(
        self,
        step: int,
        target_longitude: np.ndarray,
        target_latitude: np.ndarray,
        gamera_time: dt.datetime,
    ) -> np.ndarray:
        """Return outward current on the fixed geographic grid."""
        history, source_time = self._history(step)
        offset_seconds = abs((source_time - gamera_time).total_seconds())
        if offset_seconds > REMIX_TIME_TOLERANCE_SECONDS:
            raise RuntimeError(
                f"ReMIX Step#{step} is not aligned with GAMERA: "
                f"ReMIX={source_time.isoformat()}, GAMERA={gamera_time.isoformat()}, "
                f"offset={offset_seconds:g} s."
            )
        target_sm_lat, target_sm_lon = _geographic_grid_in_sm(
            target_latitude, target_longitude, gamera_time
        )
        north = self._hemisphere(
            "NORTH", self._fac(history, "NORTH"), target_sm_lon, target_sm_lat
        )
        south = self._hemisphere(
            "SOUTH", self._fac(history, "SOUTH"), target_sm_lon, target_sm_lat
        )
        upward_fac = _combine_remix_hemispheres(south, north)
        return _upward_fac_to_radial_current(upward_fac, target_sm_lat)
