"""Background main-field models and magnetic coordinates."""

from datetime import datetime, timedelta
from functools import partial

import apexpy
import dipole
import numpy as np
import ppigrf

from pynamit.coordinates import wrap_longitude_180
from pynamit.math.constants import RE
from pynamit.geomagnetism.kaiju_geopack import (
    kaiju_geopack_alignment,
    kaiju_geopack_dipole,
    kaiju_geopack_sm,
)


MAIN_FIELD_KINDS = ("radial", "dipole", "kaiju_dipole", "igrf")
_DIPOLE_KINDS = frozenset({"dipole", "kaiju_dipole"})


def normalize_main_field_kind(kind: str) -> str:
    """Return the canonical background-field model name."""
    normalized = str(kind).strip().lower()
    if normalized not in MAIN_FIELD_KINDS:
        raise ValueError(f"main_field_kind must be one of {list(MAIN_FIELD_KINDS)}.")
    return normalized


def is_dipole_kind(kind):
    """Return whether a kind uses centered-dipole geometry."""
    return str(kind).lower() in _DIPOLE_KINDS


def _datetime_from_decimal_year(epoch):
    """Convert a decimal year to a datetime."""
    epoch = float(epoch)
    year = int(np.floor(epoch))
    year_start = datetime(year, 1, 1, 0, 0)
    next_year_start = datetime(year + 1, 1, 1, 0, 0)
    year_seconds = (next_year_start - year_start).total_seconds()
    return year_start + timedelta(seconds=(epoch - year) * year_seconds)


def decimal_year(epoch):
    """Convert a datetime-like epoch to decimal year."""
    if not isinstance(epoch, datetime):
        return float(epoch)
    year_start = datetime(epoch.year, 1, 1, 0, 0)
    next_year_start = datetime(epoch.year + 1, 1, 1, 0, 0)
    return (
        epoch.year
        + (epoch - year_start).total_seconds() / (next_year_start - year_start).total_seconds()
    )


def _dipole_for_epoch(epoch, B0=None):
    """Return an epoch-aligned dipole."""
    base = dipole.Dipole(epoch)
    if B0 is None:
        return base
    return dipole.Dipole(dipole_pole=tuple(base.north_pole), B0=float(B0) * 1e9)


def _kaiju_dipole_for_epoch(epoch, B0=None):
    """Return a Kaiju/Geopack-aligned dipole."""
    return kaiju_geopack_dipole(_datetime_from_decimal_year(epoch), B0=B0)


def _east_north_up_to_spherical(vector):
    """Convert east/north/up components to radial/theta/phi."""
    vector = np.asarray(vector)
    return np.stack((vector[2], -vector[1], vector[0]))


def _dipole_field_components(model, r, theta, phi):
    """Evaluate a centered-dipole field in spherical components."""
    Bn, Br = model.B(90 - theta, r * 1e-3)
    return (Br * 1e-9, -Bn * 1e-9, Bn * 0)


def _igrf_field_components(epoch, r, theta, phi):
    """Evaluate IGRF in spherical components."""
    Br, Btheta, Bphi = ppigrf.igrf_gc(r * 1e-3, theta, phi, epoch)
    return (Br * 1e-9, Btheta * 1e-9, Bphi * 1e-9)


def _radial_field_components(B0, r, theta, phi):
    """Evaluate an inverse-square radial field."""
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return ((RE / r) ** 2 * B0, r * 0, r * 0)


def _normalize_field_strength(kind, B0):
    """Validate and normalize an optional reference field strength."""
    if B0 is None:
        return None
    B0 = float(B0)
    if not np.isfinite(B0) or B0 <= 0.0:
        raise ValueError("B0 must be finite and greater than zero.")
    if kind == "igrf":
        raise ValueError("B0 is not supported for the IGRF main-field model.")
    return B0


class MainField:
    """Class for representing the main magnetic field.

    Provides implementations of different magnetic field models,
    providing field components, coordinate mapping, basis vectors, and
    explicit conversions between geographic and model coordinates.

    Available models:

    - dipole: Centered dipole magnetic field using IGRF coefficients for
      alignment. The moment can be overridden with ``B0``.
    - kaiju_dipole: Centered dipole aligned with the degree-1 IGRF
      coefficients embedded in Kaiju's Geopack implementation.
      Coordinates are SM, so geographic conversion requires an
      event time. The moment can be overridden with ``B0``.
    - igrf: International Geomagnetic Reference Field in geocentric
      coordinates (with geodetic conversion ignored).
    - radial: Radial field lines with configurable magnitude.

    Notes
    -----
    The models use different horizontal coordinate systems: ``dipole``
    uses centered-dipole magnetic coordinates, ``kaiju_dipole`` uses
    Kaiju/Geopack SM coordinates, and ``igrf``/``radial`` use
    geocentric geographic coordinates. For IGRF, geodetic height is
    approximated as ``h = r - RE``.
    """

    def __init__(self, kind="dipole", epoch=2020, ionosphere_height_km=0.0, B0=None):
        """Initialize a MainField instance.

        Parameters
        ----------
        kind : {'dipole', 'kaiju_dipole', 'igrf', 'radial'}, optional
            Type of magnetic field model.
        epoch : float, optional
            Decimal year for field coefficients.
        ionosphere_height_km : float, optional
            Ionospheric height in km.
        B0 : float, optional
            Equatorial ground field magnitude for dipole/radial models
            in Tesla. If None, uses reference field magnitude for epoch.
        """
        self.kind = normalize_main_field_kind(kind)
        self.epoch = float(epoch)
        self.ionosphere_height_km = float(ionosphere_height_km)
        if not np.isfinite(self.epoch):
            raise ValueError("epoch must be finite.")
        if not np.isfinite(self.ionosphere_height_km):
            raise ValueError("ionosphere_height_km must be finite.")
        B0 = _normalize_field_strength(self.kind, B0)

        if is_dipole_kind(self.kind):
            if self.kind == "kaiju_dipole":
                self.dipole = _kaiju_dipole_for_epoch(self.epoch, B0=B0)
            else:
                self.dipole = _dipole_for_epoch(self.epoch, B0=B0)
            self._evaluate_components = partial(_dipole_field_components, self.dipole)

        elif self.kind == "igrf":
            self.apex = apexpy.Apex(self.epoch, refh=self.ionosphere_height_km)
            epoch_datetime = _datetime_from_decimal_year(self.epoch)
            self._evaluate_components = partial(_igrf_field_components, epoch_datetime)

        elif self.kind == "radial":
            # Use Dipole B0 as default.
            B0 = dipole.Dipole(self.epoch).B0 * 1e-9 if B0 is None else B0
            self._evaluate_components = partial(_radial_field_components, B0)

        else:
            raise RuntimeError("unreachable main-field kind")

    @property
    def horizontal_coordinate_system(self):
        """Return this model's horizontal coordinate system."""
        if self.kind == "kaiju_dipole":
            return "SM"
        if self.kind == "dipole":
            return "centered_dipole_magnetic"
        return "geographic"

    @property
    def geographic_transform_requires_event_time(self):
        """Return whether GEO/model conversion needs a time."""
        return self.kind == "kaiju_dipole"

    def _require_event_time(self, event_time):
        """Return an event time, raising if required."""
        if event_time is None and self.geographic_transform_requires_event_time:
            raise ValueError(
                "kaiju_dipole geographic conversion requires event_time because "
                "SM longitude is Sun-aligned."
            )
        return event_time

    @staticmethod
    def _has_tangent_vector(east, north):
        """Return whether tangent-vector components were supplied."""
        if (east is None) != (north is None):
            raise ValueError("east and north vector components must be provided together.")
        return east is not None

    def geo_to_model_coordinates(self, lat, lon, east=None, north=None, *, event_time=None):
        """Convert geographic coordinates to model coordinates.

        Parameters
        ----------
        lat, lon : array-like
            Geocentric geographic latitude and longitude in degrees.
        east, north : array-like, optional
            Tangential vector components in geographic east/north basis.
        event_time : datetime, optional
            Required for ``kaiju_dipole`` because SM longitude depends
            on the Sun-Earth geometry at the event time.

        Returns
        -------
        lat_model, lon_model[, east_model, north_model]
            Coordinates in the horizontal system returned by
            :attr:`horizontal_coordinate_system`. Longitudes are
            wrapped to [-180, 180).
        """
        self._require_event_time(event_time)
        has_vector = self._has_tangent_vector(east, north)

        if self.kind == "kaiju_dipole":
            result = kaiju_geopack_sm(event_time).geo2sm(lat, lon, east, north)
        elif self.kind == "dipole":
            if not has_vector:
                result = self.dipole.geo2mag(lat, lon)
            else:
                result = self.dipole.geo2mag(lat, lon, east, north)
        else:
            if not has_vector:
                result = (lat, lon)
            else:
                result = (lat, lon, east, north)

        result = tuple(np.asarray(value) for value in result)
        return (result[0], wrap_longitude_180(result[1]), *result[2:])

    def model_to_geo_coordinates(self, lat, lon, east=None, north=None, *, event_time=None):
        """Convert model coordinates to geographic coordinates."""
        self._require_event_time(event_time)
        has_vector = self._has_tangent_vector(east, north)

        if self.kind == "kaiju_dipole":
            result = kaiju_geopack_sm(event_time).sm2geo(lat, lon, east, north)
        elif self.kind == "dipole":
            if not has_vector:
                result = self.dipole.mag2geo(lat, lon)
            else:
                result = self.dipole.mag2geo(lat, lon, east, north)
        else:
            if not has_vector:
                result = (lat, lon)
            else:
                result = (lat, lon, east, north)

        result = tuple(np.asarray(value) for value in result)
        return (result[0], wrap_longitude_180(result[1]), *result[2:])

    def magnetic_latitude(self, r, theta, phi):
        """Return centered-dipole or apex magnetic latitude."""
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        if self.kind == "radial":
            return np.full(r.shape, np.nan, dtype=float)
        if is_dipole_kind(self.kind):
            return 90.0 - theta
        latitude, _ = self.apex.geo2apex(90.0 - theta, phi, (r - RE) * 1e-3)
        return np.asarray(latitude)

    def magnetic_latitude_trace_to_geographic(
        self, magnetic_latitude, magnetic_longitude=None, *, event_time=None, n_points=721
    ):
        """Return a magnetic-latitude trace in GEO coordinates.

        For centered dipoles, magnetic latitude means the model latitude
        returned by :meth:`geo_to_model_coordinates`. For IGRF, it means
        apex latitude at this main-field reference height.
        """
        if magnetic_longitude is None:
            magnetic_longitude = np.linspace(-180.0, 180.0, int(n_points))
        mlon = np.asarray(magnetic_longitude, dtype=float)
        mlat = np.full_like(mlon, float(magnetic_latitude), dtype=float)

        if self.kind == "radial":
            return (
                np.full_like(mlon, np.nan, dtype=float),
                np.full_like(mlon, np.nan, dtype=float),
            )
        if self.kind == "igrf":
            geo_lat, geo_lon, _ = self.apex.apex2geo(mlat, mlon, self.apex.refh)
            return np.asarray(geo_lat, dtype=float), wrap_longitude_180(geo_lon)

        geo_lat, geo_lon = self.model_to_geo_coordinates(mlat, mlon, event_time=event_time)
        return np.asarray(geo_lat, dtype=float), wrap_longitude_180(geo_lon)

    def local_time_longitude_to_model_longitude(
        self, local_time_lon, event_time, *, local_noon_longitude=0.0
    ):
        """Convert local-time longitude to model longitude.

        REMIX polar grids place noon at raw longitude zero. In
        ``kaiju_dipole`` this is already the SM longitude origin. In
        legacy ``dipole`` mode it is converted through the dipole
        package's MLT convention.
        """
        if not is_dipole_kind(self.kind):
            raise ValueError(
                "local-time longitude conversion is only defined for centered-dipole models."
            )
        local_time_lon = np.asarray(local_time_lon, dtype=float)
        if self.kind == "kaiju_dipole":
            return wrap_longitude_180(local_time_lon - local_noon_longitude)
        mlt = ((local_time_lon - local_noon_longitude) / 15.0 + 12.0) % 24.0
        return wrap_longitude_180(self.dipole.mlt2mlon(mlt, event_time))

    def alignment_metadata(self, event_time=None):
        """Return centered-field alignment metadata."""
        if not is_dipole_kind(self.kind):
            return {
                "main_field_kind": self.kind,
                "main_field_horizontal_coordinate_system": self.horizontal_coordinate_system,
            }
        event_time = self._require_event_time(event_time)
        if self.kind == "kaiju_dipole":
            dipole_model = kaiju_geopack_dipole(event_time)
            alignment = kaiju_geopack_alignment(event_time)
            noon_mlon = 0.0
            alignment["dipole_mag_noon_mlon_deg"] = float(
                np.asarray(wrap_longitude_180(dipole_model.mlt2mlon(12.0, event_time)))
            )
        else:
            dipole_model = self.dipole
            noon_mlon = dipole_model.mlt2mlon(12.0, event_time)
            alignment = {
                "dipole_alignment_model": "klaundal_dipole_igrf_centered_dipole",
                "dipole_alignment_epoch": self.epoch,
                "dipole_axis_geo_cartesian": np.asarray(dipole_model.axis, dtype=float),
                "dipole_north_pole_geo_lat_lon": np.asarray(dipole_model.north_pole, dtype=float),
                "dipole_south_pole_geo_lat_lon": np.asarray(dipole_model.south_pole, dtype=float),
            }
        return {
            "main_field_kind": self.kind,
            "main_field_horizontal_coordinate_system": self.horizontal_coordinate_system,
            "axis_geo_cartesian": np.asarray(dipole_model.axis, dtype=float),
            "north_pole_geo_lat_lon": np.asarray(dipole_model.north_pole, dtype=float),
            "south_pole_geo_lat_lon": np.asarray(dipole_model.south_pole, dtype=float),
            "noon_mlon_deg": float(np.asarray(wrap_longitude_180(noon_mlon))),
            **alignment,
        }

    def field_components(self, r, theta, phi):
        """Calculate magnetic field components.

        Parameters
        ----------
        r : array-like
            Radius in meters.
        theta : array-like
            Colatitude in degrees.
        phi : array-like
            Longitude in degrees.

        Returns
        -------
        Br : ndarray
            Radial component of the magnetic field.
        Btheta : ndarray
            Southward component of the magnetic field.
        Bphi : ndarray
            Eastward component of the magnetic field.

        Notes
        -----
        Arrays are broadcast to common shape. Output components are in
        spherical coordinate basis.
        """
        r, theta, phi = np.broadcast_arrays(r, theta, phi)
        components = self._evaluate_components(r, theta, phi)
        shaped_components = []
        for component in components:
            values = np.asarray(component)
            if values.size == 1 and r.size != 1:
                values = np.broadcast_to(values, r.shape)
            elif values.size != r.size:
                raise ValueError(
                    f"{self.kind} field component has shape {values.shape}, "
                    f"which cannot represent broadcast shape {r.shape}."
                )
            shaped_components.append(values.reshape(r.shape))
        return tuple(shaped_components)

    def inclination_sine(self, r, theta, phi):
        """Calculate sine of the inclination angle.

        Defined as the angle of the magnetic field with nadir.
        Broadcasting rules apply.

        Parameters
        ----------
        r : array-like
            Radius [m] of the points where the magnetic field is to be
            evaluated.
        theta : array-like
            Colatitude [deg] of the points where the magnetic field is
            to be evaluated.
        phi : array-like
            Longitude [deg] of the points where the magnetic field is
            to be evaluated.

        Returns
        -------
        sinI : array
            sin(inclination).
        """
        B = np.stack(self.field_components(r, theta, phi), axis=0)

        return -B[0] / np.linalg.norm(B, axis=0)

    def map_along_field_lines(self, r_dest, r, theta, phi):
        """Map coordinates along field lines.

        Maps points to new radius along magnetic field lines.

        Parameters
        ----------
        r_dest : float
            Destination radius in meters.
        r : array-like
            Starting radius in meters.
        theta : array-like
            Starting colatitude in degrees.
        phi : array-like
            Starting longitude in degrees.

        Returns
        -------
        theta_out : ndarray
            Mapped colatitude in degrees.
        phi_out : ndarray
            Mapped longitude in degrees.

        Notes
        -----
        Implementation differs by model type:
        - IGRF: Uses apex coordinates.
        - Dipole: Uses analytic dipole field line equation.
        - Radial: Angular coordinates unchanged.
        """
        r, theta, phi = np.broadcast_arrays(r, theta, phi)

        if self.kind == "radial":
            # Angular coordinates are kept the same.
            theta_out = theta
            phi_out = phi
        elif is_dipole_kind(self.kind):
            # Map from r to r_dest for dipole field.
            hemisphere = np.sign(90 - theta)
            unsigned_magnetic_latitude = 90 - np.rad2deg(
                np.arcsin(np.sin(np.deg2rad(theta)) * np.sqrt(r_dest / r))
            )
            theta_out = 90 - hemisphere * unsigned_magnetic_latitude
            # Longitude is kept the same.
            phi_out = phi

        elif self.kind == "igrf":
            # Use apexpy to map along IGRF field lines.
            mlat, mlon = self.apex.geo2apex(90 - theta, phi, (r - RE) * 1e-3)
            lat_out, phi_out, _ = self.apex.apex2geo(mlat, mlon, (r_dest - RE) * 1e-3)
            theta_out = 90 - lat_out

        return (theta_out, phi_out)

    def conjugate_coordinates(self, r, theta, phi):
        """Find magnetically conjugate points.

        Calculates coordinates of magnetically connected points
        in opposite hemisphere.

        Parameters
        ----------
        r : array-like
            Radius in meters.
        theta : array-like
            Colatitude in degrees.
        phi : array-like
            Longitude in degrees.

        Returns
        -------
        theta_conj : ndarray
            Conjugate point colatitude in degrees.
        phi_conj : ndarray
            Conjugate point longitude in degrees.

        Raises
        ------
        ValueError
            If called with radial field model.

        Notes
        -----
        Implementation differs by model type:
        - IGRF: Uses apex coordinate transformations.
        - Dipole: Conjugate points are at (180°-`theta`, `phi`).
        - Radial: Not defined.
        """
        r, theta, phi = np.broadcast_arrays(r, theta, phi)

        if self.kind == "radial":
            raise ValueError("Conjugate coordinates do not exist with radial field lines")

        if is_dipole_kind(self.kind):
            theta_conj, phi_conj = (180 - theta, phi)
        elif self.kind == "igrf":
            h = (r - RE) * 1e-3
            mlat, mlon = self.apex.geo2apex(90 - theta, phi, h)
            glat, phi_conj, _ = self.apex.apex2geo(-mlat, mlon, h)
            theta_conj = 90 - glat

        return (theta_conj, phi_conj)

    def basis_vectors(self, r, theta, phi):
        """Calculate apex coordinate basis vectors.

        Computes modified apex coordinate basis vectors defined in
        Richmond (1995).

        Parameters
        ----------
        r : array-like
            Radius in meters.
        theta : array-like
            Colatitude in degrees.
        phi : array-like
            Longitude in degrees.

        Returns
        -------
        d1, d2, d3 : ndarray
            Contravariant basis vectors, shape (3,N).
        e1, e2, e3 : ndarray
            Covariant basis vectors, shape (3,N).

        Notes
        -----
        Vector components are in spherical coordinates.

        Implementation differs by model type:
        - IGRF: Uses apex coordinate transformations.
        - Dipole: Uses analytic dipole expressions.
        - Radial: Uses simple orthonormal vectors.
        """
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        if self.kind == "radial":
            size = r.size
            zeros = np.zeros(size)
            ones = np.ones(size)
            east = np.vstack((zeros, zeros, ones))
            north = np.vstack((zeros, -ones, zeros))
            upward = np.vstack((ones, zeros, zeros))
            field_sign = np.sign(np.asarray(self.field_components(r, theta, phi)[0])).reshape(
                1, -1
            )
            return (east, north * field_sign, upward * field_sign) * 2

        if is_dipole_kind(self.kind):
            vectors = self.dipole.get_apex_base_vectors(90 - theta, r * 1e-3, R=RE * 1e-3)
        else:
            vectors = self.apex.basevectors_apex(90 - theta, phi, (r - RE) * 1e-3, coords="geo")[
                6:
            ]

        return tuple(_east_north_up_to_spherical(vector) for vector in vectors)

    def magnetic_colatitude_at_longitude(self, longitude, magnetic_colatitude=90):
        """Return a magnetic-colatitude trace sampled by longitude.

        Parameters
        ----------
        longitude : array-like
            Longitudes in degrees.
        magnetic_colatitude : float, optional
            Magnetic colatitude of the trace in degrees.

        Returns
        -------
        array
            Geographic/model colatitude at each requested longitude.
        """
        longitude = np.asarray(longitude, dtype=float) % 360

        if self.kind == "radial":
            return np.full_like(longitude, np.nan, dtype=float)

        if is_dipole_kind(self.kind):
            return np.zeros_like(longitude) + magnetic_colatitude

        if self.kind == "igrf":
            mlon = np.linspace(0, 360, 360)
            # Calculate latitude of evenly spaced points.
            lat, lon, _ = self.apex.apex2geo(90 - magnetic_colatitude, mlon, self.apex.refh)
            return np.interp(longitude.reshape(-1), lon % 360, 90 - lat, period=360).reshape(
                longitude.shape
            )
