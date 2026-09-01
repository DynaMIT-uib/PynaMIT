"""Coordinate contexts for map plots."""

from dataclasses import dataclass

import numpy as np
from kompe import SphericalGrid

from pynamit.coordinates import (
    DEFAULT_LOCAL_TIME_GRID_HOURS,
    local_noon_longitude,
    longitude_to_local_time_from_noon_longitude,
    wrap_longitude_180,
)
from pynamit.geodesy import library_geographic_to_spherical_geo

_VALID_LONGITUDE_KINDS = {"geographic", "magnetic"}
_VALID_LOCAL_TIME_KINDS = {"solar", "magnetic"}


def regular_geographic_grid(nlat=60, nlon=100, lat_range=(-89.9, 89.9), lon_range=(-180.0, 180.0)):
    """Return latitude, longitude, and a regular geographic grid."""
    latitude = np.linspace(lat_range[0], lat_range[1], int(nlat))
    longitude = np.linspace(lon_range[0], lon_range[1], int(nlon))
    longitude, latitude = np.meshgrid(longitude, latitude)
    return latitude, longitude, SphericalGrid(lat=latitude, lon=longitude)


def model_grid_from_geographic(main_field, latitude, longitude):
    """Return model coordinates underlying a geographic grid."""
    model_latitude, model_longitude = main_field.geo_to_model_coordinates(latitude, longitude)
    return SphericalGrid(lat=model_latitude, lon=model_longitude)


def _as_float_scalar(value, name):
    """Return a scalar float from a scalar-like value."""
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar-like.")
    return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class MapCoordinateContext:
    """Local-time convention for a plotted longitude coordinate.

    The context does not transform whole grids. It records the longitude
    coordinate used by the plot and the local-time convention used to
    label that coordinate.
    """

    noon_longitude: float
    longitude_kind: str = "geographic"
    local_time_kind: str = "solar"
    label: str | None = None
    reference_time: object | None = None

    def __post_init__(self):
        """Normalize and validate the context metadata."""
        longitude_kind = str(self.longitude_kind).lower()
        local_time_kind = str(self.local_time_kind).lower()
        if longitude_kind not in _VALID_LONGITUDE_KINDS:
            raise ValueError(f"longitude_kind must be one of {_VALID_LONGITUDE_KINDS}.")
        if local_time_kind not in _VALID_LOCAL_TIME_KINDS:
            raise ValueError(f"local_time_kind must be one of {_VALID_LOCAL_TIME_KINDS}.")
        label = self.label
        if label is None:
            label = "MLT" if local_time_kind == "magnetic" else "LT"
        object.__setattr__(self, "noon_longitude", wrap_longitude_180(self.noon_longitude))
        object.__setattr__(self, "longitude_kind", longitude_kind)
        object.__setattr__(self, "local_time_kind", local_time_kind)
        object.__setattr__(self, "label", str(label))

    @classmethod
    def from_noon_longitude(
        cls,
        noon_longitude,
        *,
        longitude_kind="geographic",
        local_time_kind="solar",
        label=None,
        reference_time=None,
    ):
        """Create a context from an explicit local-noon longitude."""
        return cls(
            noon_longitude=float(noon_longitude),
            longitude_kind=longitude_kind,
            local_time_kind=local_time_kind,
            label=label,
            reference_time=reference_time,
        )

    @classmethod
    def geographic(cls, reference_time):
        """Create a geographic mean-solar-local-time context."""
        return cls(
            noon_longitude=local_noon_longitude(reference_time),
            longitude_kind="geographic",
            local_time_kind="solar",
            label="LT",
            reference_time=reference_time,
        )

    @classmethod
    def magnetic(cls, reference_time, dipole, *, apex=None, apex_height=0.0):
        """Create a magnetic-local-time context.

        Without ``apex``, the context longitude is magnetic longitude.
        With ``apex``, the magnetic noon meridian is converted to a
        geographic longitude for global geographic maps.
        """
        magnetic_noon = _as_float_scalar(
            dipole.mlt2mlon(12, reference_time), "magnetic noon longitude"
        )
        if apex is None:
            return cls(
                noon_longitude=magnetic_noon,
                longitude_kind="magnetic",
                local_time_kind="magnetic",
                label="MLT",
                reference_time=reference_time,
            )
        library_latitude, library_longitude, _ = apex.apex2geo(0, magnetic_noon, apex_height)
        _, geographic_noon = library_geographic_to_spherical_geo(
            library_latitude, library_longitude
        )
        return cls(
            noon_longitude=_as_float_scalar(geographic_noon, "geographic noon longitude"),
            longitude_kind="geographic",
            local_time_kind="magnetic",
            label="MLT",
            reference_time=reference_time,
        )

    def projection(self):
        """Return PlateCarree centered on this context's noon."""
        import cartopy.crs as ccrs

        return ccrs.PlateCarree(central_longitude=self.noon_longitude)

    def longitude_to_local_time(self, lon, *, wrap=True):
        """Convert plotted longitude to local-time hours."""
        return longitude_to_local_time_from_noon_longitude(lon, self.noon_longitude, wrap=wrap)

    def local_time_to_longitude(self, local_time_hours):
        """Convert local-time hours to plotted longitude."""
        return wrap_longitude_180(
            self.noon_longitude + (np.asarray(local_time_hours, dtype=float) - 12.0) * 15.0
        )

    def local_time_grid_longitudes(self, hours=DEFAULT_LOCAL_TIME_GRID_HOURS):
        """Return plotted longitudes for selected local-time ticks."""
        return self.local_time_to_longitude(np.asarray(hours, dtype=float))

    def local_time_longitude_to_coordinate(self, lon, *, local_noon_longitude=0.0):
        """Convert source LT-like longitude to plotted longitude."""
        return wrap_longitude_180(
            np.asarray(lon, dtype=float) - float(local_noon_longitude) + self.noon_longitude
        )

    def format_local_time_label(self, lon, pos=None):
        """Format a longitude tick as a local-time label."""
        del pos
        hour = int(np.round(self.longitude_to_local_time(lon))) % 24
        return f"{hour} {self.label}"

    def local_time_formatter(self):
        """Create a Matplotlib formatter for this local-time context."""
        from matplotlib.ticker import FuncFormatter

        return FuncFormatter(self.format_local_time_label)

    def apply_grid_labels(self, gridliner, *, hours=DEFAULT_LOCAL_TIME_GRID_HOURS):
        """Apply this context's LT ticks to a Cartopy gridliner."""
        from matplotlib.ticker import FixedLocator

        gridliner.xlocator = FixedLocator(self.local_time_grid_longitudes(hours))
        gridliner.xformatter = self.local_time_formatter()
        return gridliner


__all__ = ["MapCoordinateContext", "model_grid_from_geographic", "regular_geographic_grid"]
