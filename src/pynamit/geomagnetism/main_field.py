"""Background main-field models and magnetic coordinates."""

from datetime import datetime, timezone

import apexpy
import dipole
import numpy as np
import ppigrf
from kompe.constants import EARTH_RADIUS_M

from pynamit.coordinates import (
    CENTERED_DIPOLE,
    GEOCENTRIC_GEOGRAPHIC,
    decimal_year_to_datetime,
    wrap_longitude_180,
)
from pynamit.coordinates import local_noon_longitude as geographic_noon_longitude
from pynamit.geodesy import (
    library_geographic_to_spherical_geo,
    spherical_geo_to_library_geographic,
)
from pynamit.geomagnetism.kaiju_geopack import (
    kaiju_geopack_alignment,
    kaiju_geopack_dipole,
    kaiju_geopack_mag,
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


def horizontal_coordinate_system_for_kind(kind):
    """Return the canonical horizontal frame for a main-field kind."""
    normalized = normalize_main_field_kind(kind)
    return CENTERED_DIPOLE if normalized == "dipole" else GEOCENTRIC_GEOGRAPHIC


def decimal_year(epoch):
    """Convert a datetime-like epoch to decimal year."""
    if not isinstance(epoch, datetime):
        return float(epoch)
    if epoch.tzinfo is not None:
        epoch = epoch.astimezone(timezone.utc).replace(tzinfo=None)
    year_start = datetime(epoch.year, 1, 1, 0, 0)
    next_year_start = datetime(epoch.year + 1, 1, 1, 0, 0)
    return (
        epoch.year
        + (epoch - year_start).total_seconds() / (next_year_start - year_start).total_seconds()
    )


def _east_north_up_to_spherical(vector):
    """Convert east/north/up components to radial/theta/phi."""
    vector = np.asarray(vector)
    return np.stack((vector[2], -vector[1], vector[0]))


def _igrf_apex_input_from_spherical(r, theta, phi):
    """Return the spherical approximation used by ApexPy."""
    return spherical_geo_to_library_geographic(
        90.0 - np.asarray(theta, dtype=float),
        phi,
        (np.asarray(r, dtype=float) - EARTH_RADIUS_M) * 1e-3,
    )


class MainField:
    """Class for representing the main magnetic field.

    Provides implementations of different magnetic field models,
    providing field components, coordinate mapping, basis vectors, and
    explicit conversions between geographic and model coordinates.

    Available models:

    - dipole: Centered dipole magnetic field using IGRF coefficients for
      alignment. The moment can be overridden with ``B0``.
    - kaiju_dipole: Centered dipole aligned with the degree-1 IGRF
      coefficients embedded in Kaiju's Geopack implementation. Public
      coordinates are geographic; MAG is used only inside the magnetic
      model. The moment can be overridden with ``B0``.
    - igrf: International Geomagnetic Reference Field in geocentric
      coordinates (with geodetic conversion ignored).
    - radial: Radial field lines with configurable magnitude.

    Notes
    -----
    ``kaiju_dipole``, ``igrf``, and ``radial`` use geocentric geographic
    coordinates. The generic ``dipole`` model retains centered-dipole
    magnetic coordinates. SM and MAG are physical working frames owned
    by the Kaiju field/source adapters, not simulation-state coordinate
    systems. For IGRF, geodetic height is approximated as
    ``h = r - EARTH_RADIUS_M``.
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
        self.epoch_datetime = decimal_year_to_datetime(self.epoch)
        if B0 is not None:
            B0 = float(B0)
            if not np.isfinite(B0) or B0 <= 0.0:
                raise ValueError("B0 must be finite and greater than zero.")
            if self.kind == "igrf":
                raise ValueError("B0 is not supported for the IGRF main-field model.")
        self.B0 = B0

        if is_dipole_kind(self.kind):
            if self.kind == "kaiju_dipole":
                self.dipole = kaiju_geopack_dipole(self.epoch_datetime, B0=B0)
                self._mag_transform = kaiju_geopack_mag(self.epoch_datetime)
            else:
                self.dipole = dipole.Dipole(self.epoch)
                if B0 is not None:
                    self.dipole = dipole.Dipole(
                        dipole_pole=tuple(self.dipole.north_pole), B0=B0 * 1e9
                    )
            self.B0 = self.dipole.B0 * 1e-9

        elif self.kind == "igrf":
            self.apex = apexpy.Apex(self.epoch, refh=self.ionosphere_height_km)

        else:
            # Use Dipole B0 as default.
            self.B0 = dipole.Dipole(self.epoch).B0 * 1e-9 if B0 is None else B0

    @property
    def horizontal_coordinate_system(self):
        """Return this model's horizontal coordinate system."""
        return horizontal_coordinate_system_for_kind(self.kind)

    def __repr__(self):
        """Summarize the background magnetic-field model."""
        return (
            f"MainField(kind={self.kind!r}, epoch={self.epoch:g}, "
            f"ionosphere_height_km={self.ionosphere_height_km:g}, B0={self.B0!r})"
        )

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
            Accepted for a uniform coordinate API. Frame orientation is
            fixed by the main-field epoch, so transforms do not use it.

        Returns
        -------
        lat_model, lon_model[, east_model, north_model]
            Coordinates in the horizontal system returned by
            :attr:`horizontal_coordinate_system`. Longitudes are
            wrapped to [-180, 180).
        """
        has_vector = self._has_tangent_vector(east, north)

        if self.kind == "dipole":
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
        has_vector = self._has_tangent_vector(east, north)

        if self.kind == "dipole":
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
        if self.kind == "kaiju_dipole":
            magnetic_latitude, _ = self._mag_transform.geo2mag(90.0 - theta, phi)
            return np.asarray(magnetic_latitude)
        if self.kind == "dipole":
            return 90.0 - theta
        library_latitude, library_longitude, library_height = _igrf_apex_input_from_spherical(
            r, theta, phi
        )
        latitude, _ = self.apex.geo2apex(library_latitude, library_longitude, library_height)
        return np.asarray(latitude)

    def magnetic_latitude_trace_to_geographic(
        self, magnetic_latitude, magnetic_longitude=None, *, n_points=721
    ):
        """Return a magnetic-latitude trace in GEO coordinates.

        For centered dipoles these are conventional MAG coordinates.
        For IGRF they are apex coordinates at this field's reference
        height.
        """
        if magnetic_longitude is None:
            magnetic_longitude = np.linspace(-180.0, 180.0, int(n_points))
        mlat, mlon = np.broadcast_arrays(
            np.asarray(magnetic_latitude, dtype=float), np.asarray(magnetic_longitude, dtype=float)
        )

        return self.magnetic_to_geographic_coordinates(mlat, mlon)

    def magnetic_to_geographic_coordinates(self, latitude, longitude):
        """Convert this field model's magnetic coordinates to GEO."""
        latitude, longitude = np.broadcast_arrays(
            np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
        )

        if self.kind == "radial":
            return (
                np.full_like(latitude, np.nan, dtype=float),
                np.full_like(longitude, np.nan, dtype=float),
            )
        if self.kind == "igrf":
            library_latitude, library_longitude, _ = self.apex.apex2geo(
                latitude, longitude, self.apex.refh
            )
            geo_lat, geo_lon = library_geographic_to_spherical_geo(
                library_latitude, library_longitude
            )
            return np.asarray(geo_lat, dtype=float), wrap_longitude_180(geo_lon)
        if self.kind == "kaiju_dipole":
            geo_lat, geo_lon = self._mag_transform.mag2geo(latitude, longitude)
        else:
            geo_lat, geo_lon = self.model_to_geo_coordinates(latitude, longitude)
        return np.asarray(geo_lat, dtype=float), wrap_longitude_180(geo_lon)

    def geographic_to_magnetic_coordinates(self, latitude, longitude):
        """Convert GEO to this field model's magnetic coordinates."""
        latitude, longitude = np.broadcast_arrays(
            np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
        )

        if self.kind == "radial":
            return (
                np.full_like(latitude, np.nan, dtype=float),
                np.full_like(longitude, np.nan, dtype=float),
            )
        if self.kind == "igrf":
            library_latitude, library_longitude, library_height = (
                spherical_geo_to_library_geographic(latitude, longitude, self.apex.refh)
            )
            magnetic_latitude, magnetic_longitude = self.apex.geo2apex(
                library_latitude, library_longitude, library_height
            )
        elif self.kind == "kaiju_dipole":
            magnetic_latitude, magnetic_longitude = self._mag_transform.geo2mag(
                latitude, longitude
            )
        else:
            magnetic_latitude, magnetic_longitude = self.geo_to_model_coordinates(
                latitude, longitude
            )
        return np.asarray(magnetic_latitude, dtype=float), wrap_longitude_180(magnetic_longitude)

    def magnetic_noon_longitude(self, event_time):
        """Return the local-noon meridian in magnetic coordinates."""
        if self.kind == "radial":
            raise ValueError("magnetic noon is not defined for a radial field model.")
        if self.kind == "igrf":
            if event_time is None:
                raise ValueError("IGRF magnetic noon requires event_time.")
            return float(np.asarray(wrap_longitude_180(self.apex.mlt2mlon(12.0, event_time))))
        if self.kind == "kaiju_dipole":
            if event_time is None:
                raise ValueError(
                    "kaiju_dipole magnetic noon requires event_time because SM follows the Sun."
                )
            sm = kaiju_geopack_sm(event_time)
            geo_lat, geo_lon = sm.sm2geo(0.0, 0.0)
            _, magnetic_lon = self._mag_transform.geo2mag(geo_lat, geo_lon)
            return float(np.asarray(wrap_longitude_180(magnetic_lon)))
        return float(np.asarray(wrap_longitude_180(self.dipole.mlt2mlon(12.0, event_time))))

    def local_noon_longitude(self, event_time):
        """Return the local-noon longitude in model coordinates."""
        if self.horizontal_coordinate_system == GEOCENTRIC_GEOGRAPHIC:
            return geographic_noon_longitude(event_time)
        return self.magnetic_noon_longitude(event_time)

    def alignment_metadata(self, event_time=None):
        """Return centered-field alignment metadata."""
        if not is_dipole_kind(self.kind):
            return {
                "main_field_kind": self.kind,
                "main_field_horizontal_coordinate_system": self.horizontal_coordinate_system,
            }
        if self.kind == "kaiju_dipole":
            alignment_time = self.epoch_datetime if event_time is None else event_time
            dipole_model = self.dipole
            alignment = kaiju_geopack_alignment(alignment_time, magnetic_epoch=self.epoch_datetime)
            noon_longitude = self.local_noon_longitude(alignment_time)
            alignment["magnetic_noon_longitude_deg"] = self.magnetic_noon_longitude(alignment_time)
        else:
            dipole_model = self.dipole
            noon_longitude = dipole_model.mlt2mlon(12.0, event_time)
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
            "noon_model_longitude_deg": float(np.asarray(wrap_longitude_180(noon_longitude))),
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
        field_theta = theta
        if self.kind == "kaiju_dipole":
            magnetic_latitude, magnetic_longitude = self._mag_transform.geo2mag(90.0 - theta, phi)
            field_theta = 90.0 - magnetic_latitude

        if is_dipole_kind(self.kind):
            Bnorth, Br = self.dipole.B(90.0 - field_theta, r * 1e-3)
            components = (Br * 1e-9, -Bnorth * 1e-9, Bnorth * 0.0)
        elif self.kind == "igrf":
            Br, Btheta, Bphi = ppigrf.igrf_gc(r * 1e-3, theta, phi, self.epoch_datetime)
            components = (Br * 1e-9, Btheta * 1e-9, Bphi * 1e-9)
        else:
            components = ((EARTH_RADIUS_M / r) ** 2 * self.B0, r * 0.0, r * 0.0)

        if self.kind == "kaiju_dipole":
            Br, Btheta_magnetic, Bphi_magnetic = components
            _, _, Bphi, north = self._mag_transform.mag2geo(
                magnetic_latitude, magnetic_longitude, east=Bphi_magnetic, north=-Btheta_magnetic
            )
            components = (Br, -north, Bphi)

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
        elif self.kind == "kaiju_dipole":
            magnetic_latitude, magnetic_longitude = self._mag_transform.geo2mag(90.0 - theta, phi)
            hemisphere = np.sign(magnetic_latitude)
            unsigned_magnetic_latitude = 90.0 - np.rad2deg(
                np.arcsin(np.cos(np.deg2rad(magnetic_latitude)) * np.sqrt(r_dest / r))
            )
            mapped_magnetic_latitude = hemisphere * unsigned_magnetic_latitude
            latitude_out, phi_out = self._mag_transform.mag2geo(
                mapped_magnetic_latitude, magnetic_longitude
            )
            theta_out = 90.0 - latitude_out
        elif self.kind == "dipole":
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
            library_latitude, library_longitude, library_height = _igrf_apex_input_from_spherical(
                r, theta, phi
            )
            mlat, mlon = self.apex.geo2apex(library_latitude, library_longitude, library_height)
            library_latitude_out, library_longitude_out, _ = self.apex.apex2geo(
                mlat, mlon, (np.asarray(r_dest, dtype=float) - EARTH_RADIUS_M) * 1e-3
            )
            latitude_out, phi_out = library_geographic_to_spherical_geo(
                library_latitude_out, library_longitude_out
            )
            theta_out = 90.0 - latitude_out

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

        if self.kind == "kaiju_dipole":
            magnetic_latitude, magnetic_longitude = self._mag_transform.geo2mag(90.0 - theta, phi)
            latitude_conj, phi_conj = self._mag_transform.mag2geo(
                -magnetic_latitude, magnetic_longitude
            )
            theta_conj = 90.0 - latitude_conj
        elif self.kind == "dipole":
            theta_conj, phi_conj = (180 - theta, phi)
        elif self.kind == "igrf":
            library_latitude, library_longitude, library_height = _igrf_apex_input_from_spherical(
                r, theta, phi
            )
            mlat, mlon = self.apex.geo2apex(library_latitude, library_longitude, library_height)
            library_latitude_conj, library_longitude_conj, _ = self.apex.apex2geo(
                -mlat, mlon, library_height
            )
            latitude_conj, phi_conj = library_geographic_to_spherical_geo(
                library_latitude_conj, library_longitude_conj
            )
            theta_conj = 90.0 - latitude_conj

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

        if self.kind == "kaiju_dipole":
            magnetic_latitude, magnetic_longitude = self._mag_transform.geo2mag(90.0 - theta, phi)
            vectors = self.dipole.get_apex_base_vectors(
                magnetic_latitude, r * 1e-3, R=EARTH_RADIUS_M * 1e-3
            )

            def magnetic_vector_to_geographic(vector):
                _, _, east, north = self._mag_transform.mag2geo(
                    magnetic_latitude, magnetic_longitude, east=vector[0], north=vector[1]
                )
                return np.stack((vector[2], -north, east))

            return tuple(magnetic_vector_to_geographic(vector) for vector in vectors)

        if self.kind == "dipole":
            vectors = self.dipole.get_apex_base_vectors(
                90 - theta, r * 1e-3, R=EARTH_RADIUS_M * 1e-3
            )
        else:
            library_latitude, library_longitude, library_height = _igrf_apex_input_from_spherical(
                r, theta, phi
            )
            vectors = self.apex.basevectors_apex(
                library_latitude, library_longitude, library_height, coords="geo"
            )[6:]

        return tuple(_east_north_up_to_spherical(vector) for vector in vectors)
