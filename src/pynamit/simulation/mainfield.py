"""Mainfield module.

This module contains the Mainfield class for main magnetic field
representation.
"""

import ppigrf
import apexpy
import dipole
import numpy as np
from datetime import datetime, timedelta
from pynamit.math.constants import RE
from pynamit.coordinates import wrap_longitude_180
from pynamit.simulation.kaiju_dipole import (
    kaiju_geopack_alignment,
    kaiju_geopack_dipole,
    kaiju_geopack_sm,
)


DIPOLE_KINDS = {"dipole", "kaiju_dipole"}
MAINFIELD_KINDS = ("radial", "dipole", "kaiju_dipole", "igrf")


def is_dipole_kind(kind):
    """Return whether a main-field kind uses centered-dipole geometry."""
    return str(kind).lower() in DIPOLE_KINDS


def datetime_from_decimal_year(epoch):
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
    return epoch.year + (epoch - year_start).total_seconds() / (
        next_year_start - year_start
    ).total_seconds()


def _dipole_for_epoch(epoch, B0=None):
    """Return an epoch-aligned dipole, optionally overriding magnitude."""
    base = dipole.Dipole(epoch)
    if B0 is None:
        return base
    return dipole.Dipole(dipole_pole=tuple(base.north_pole), B0=float(B0) * 1e9)


def _kaiju_dipole_for_epoch(epoch, B0=None):
    """Return a Kaiju/Geopack-aligned dipole with optional magnitude override."""
    return kaiju_geopack_dipole(datetime_from_decimal_year(epoch), B0=B0)


class Mainfield:
    """Class for representing the main magnetic field.

    Provides implementations of different magnetic field models,
    providing field components, coordinate mapping, basis vectors, and
    explicit conversions between geographic and model coordinates.

    Available models:
    - dipole: Centered dipole magnetic field using IGRF coefficients for
      alignment. The moment can be overridden with ``B0``.
    - kaiju_dipole: Centered dipole aligned with the degree-1 IGRF
      coefficients embedded in Kaiju's Geopack implementation. Coordinates
      are SM, so geographic conversion requires an event time. The moment can
      be overridden with ``B0``.
    - igrf: International Geomagnetic Reference Field in geocentric
      coordinates (with geodetic conversion ignored).
    - radial: Radial field lines with configurable magnitude.

    Attributes
    ----------
    kind : str
        Active field model type.
    dpl : dipole.Dipole
        Dipole field instance for centered-dipole kinds.
    apx : apexpy.Apex
        Apex coordinate transformer (if kind=='igrf').

    Notes
    -----
    The models use different horizontal coordinate systems: ``dipole`` uses
    centered-dipole magnetic coordinates, ``kaiju_dipole`` uses Kaiju/Geopack
    SM coordinates, and ``igrf``/``radial`` use geocentric geographic
    coordinates. For IGRF, geodetic height is approximated as ``h = r - RE``.
    """

    def __init__(self, kind="dipole", epoch=2020, hI=0.0, B0=None):
        """Initialize a Mainfield instance.

        Parameters
        ----------
        kind : {'dipole', 'kaiju_dipole', 'igrf', 'radial'}, optional
            Type of magnetic field model.
        epoch : float, optional
            Decimal year for field coefficients.
        hI : float, optional
            Ionospheric height in km.
        B0 : float, optional
            Equatorial ground field magnitude for dipole/radial models in
            Tesla. If None, uses reference field magnitude for epoch.
        """
        if kind.lower() not in MAINFIELD_KINDS:
            raise ValueError(f"kind must be one of {', '.join(MAINFIELD_KINDS)}")

        self.kind = kind.lower()
        self.epoch = float(epoch)
        self.hI = float(hI)

        # Define magnetic field and mapping functions for chosen model.
        if is_dipole_kind(self.kind):
            if self.kind == "kaiju_dipole":
                self.dpl = _kaiju_dipole_for_epoch(epoch, B0=B0)
            else:
                self.dpl = _dipole_for_epoch(epoch, B0=B0)

            def _Bfunc(r, theta, phi):
                Bn, Br = self.dpl.B(90 - theta, r * 1e-3)
                return (Br * 1e-9, -Bn * 1e-9, Bn * 0)

        elif self.kind == "igrf":
            self.apx = apexpy.Apex(float(epoch), refh=hI)
            epoch = datetime_from_decimal_year(epoch)

            def _Bfunc(r, theta, phi):
                Br, Btheta, Bphi = ppigrf.igrf_gc(r * 1e-3, theta, phi, epoch)
                return (Br * 1e-9, Btheta * 1e-9, Bphi * 1e-9)

        elif self.kind == "radial":
            # Use Dipole B0 as default.
            B0 = dipole.Dipole(epoch).B0 * 1e-9 if B0 is None else float(B0)

            def _Bfunc(r, theta, phi):
                r, theta, phi = np.broadcast_arrays(r, theta, phi)
                return ((RE / r) ** 2 * B0, r * 0, r * 0)

        else:
            raise RuntimeError("unreachable main-field kind")

        self._Bfunc = _Bfunc

    @property
    def coordinate_system(self):
        """Return the horizontal coordinate system expected by this model."""
        if self.kind == "kaiju_dipole":
            return "SM"
        if self.kind == "dipole":
            return "centered_dipole_magnetic"
        return "geographic"

    @property
    def geographic_transform_requires_event_time(self):
        """Return whether GEO/model conversion requires an explicit time."""
        return self.kind == "kaiju_dipole"

    def _require_event_time(self, event_time):
        """Return an event time, raising if the chosen model requires one."""
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
        """Convert geographic coordinates and optional vectors to model coordinates.

        Parameters
        ----------
        lat, lon : array-like
            Geocentric geographic latitude and longitude in degrees.
        east, north : array-like, optional
            Tangential vector components in the geographic east/north basis.
        event_time : datetime, optional
            Required for ``kaiju_dipole`` because SM longitude depends on the
            Sun-Earth geometry at the event time.

        Returns
        -------
        lat_model, lon_model[, east_model, north_model]
            Coordinates in the horizontal system returned by
        :attr:`coordinate_system`. Longitudes are wrapped to [-180, 180).
        """
        self._require_event_time(event_time)
        has_vector = self._has_tangent_vector(east, north)

        if self.kind == "kaiju_dipole":
            result = kaiju_geopack_sm(event_time).geo2sm(lat, lon, east, north)
        elif self.kind == "dipole":
            if not has_vector:
                result = self.dpl.geo2mag(lat, lon)
            else:
                result = self.dpl.geo2mag(lat, lon, east, north)
        else:
            if not has_vector:
                result = (lat, lon)
            else:
                result = (lat, lon, east, north)

        result = tuple(np.asarray(value) for value in result)
        return (result[0], wrap_longitude_180(result[1]), *result[2:])

    def model_to_geo_coordinates(self, lat, lon, east=None, north=None, *, event_time=None):
        """Convert model coordinates and optional vectors to geographic coordinates."""
        self._require_event_time(event_time)
        has_vector = self._has_tangent_vector(east, north)

        if self.kind == "kaiju_dipole":
            result = kaiju_geopack_sm(event_time).sm2geo(lat, lon, east, north)
        elif self.kind == "dipole":
            if not has_vector:
                result = self.dpl.mag2geo(lat, lon)
            else:
                result = self.dpl.mag2geo(lat, lon, east, north)
        else:
            if not has_vector:
                result = (lat, lon)
            else:
                result = (lat, lon, east, north)

        result = tuple(np.asarray(value) for value in result)
        return (result[0], wrap_longitude_180(result[1]), *result[2:])

    def local_time_longitude_to_model_longitude(
        self,
        local_time_lon,
        event_time,
        *,
        local_noon_longitude=0.0,
    ):
        """Convert a REMIX/MAGE local-time longitude to this model longitude.

        REMIX polar grids place noon at raw longitude zero. In ``kaiju_dipole``
        this is already the SM longitude origin. In legacy ``dipole`` mode it
        is converted through the dipole package's MLT convention.
        """
        if not is_dipole_kind(self.kind):
            raise ValueError(
                "local-time longitude conversion is only defined for centered-dipole models."
            )
        local_time_lon = np.asarray(local_time_lon, dtype=float)
        if self.kind == "kaiju_dipole":
            return wrap_longitude_180(local_time_lon - local_noon_longitude)
        mlt = ((local_time_lon - local_noon_longitude) / 15.0 + 12.0) % 24.0
        return wrap_longitude_180(self.dpl.mlt2mlon(mlt, event_time))

    def alignment_metadata(self, event_time=None):
        """Return centered-field alignment metadata for logs and output files."""
        if not is_dipole_kind(self.kind):
            return {
                "mainfield_kind": self.kind,
                "mainfield_coordinate_system": self.coordinate_system,
            }
        event_time = self._require_event_time(event_time)
        if self.kind == "kaiju_dipole":
            dpl = kaiju_geopack_dipole(event_time)
            alignment = kaiju_geopack_alignment(event_time)
            noon_mlon = 0.0
            alignment["dipole_mag_noon_mlon_deg"] = float(
                np.asarray(wrap_longitude_180(dpl.mlt2mlon(12.0, event_time)))
            )
        else:
            dpl = self.dpl
            noon_mlon = dpl.mlt2mlon(12.0, event_time)
            alignment = {
                "dipole_alignment_model": "klaundal_dipole_igrf_centered_dipole",
                "dipole_alignment_epoch": self.epoch,
                "dipole_axis_geo_cartesian": np.asarray(dpl.axis, dtype=float),
                "dipole_north_pole_geo_lat_lon": np.asarray(
                    dpl.north_pole, dtype=float
                ),
                "dipole_south_pole_geo_lat_lon": np.asarray(
                    dpl.south_pole, dtype=float
                ),
            }
        return {
            "mainfield_kind": self.kind,
            "mainfield_coordinate_system": self.coordinate_system,
            "axis_geo_cartesian": np.asarray(dpl.axis, dtype=float),
            "north_pole_geo_lat_lon": np.asarray(dpl.north_pole, dtype=float),
            "south_pole_geo_lat_lon": np.asarray(dpl.south_pole, dtype=float),
            "noon_mlon_deg": float(np.asarray(wrap_longitude_180(noon_mlon))),
            **alignment,
        }

    def get_B(self, r, theta, phi):
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
        return self._Bfunc(r, theta, phi)

    def get_sinI(self, r, theta, phi):
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
        B = np.vstack(self.get_B(r, theta, phi))

        return -B[0] / np.linalg.norm(B, axis=0)

    def map_coords(self, r_dest, r, theta, phi):
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

        if is_dipole_kind(self.kind):
            # Map from r to r_dest for dipole field.
            hemisphere = np.sign(90 - theta)
            la_ = 90 - np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta)) * np.sqrt(r_dest / r)))
            theta_out = 90 - hemisphere * la_
            # Longitude is kept the same.
            phi_out = phi

        elif self.kind == "igrf":
            # Use apexpy to map along IGRF field lines.
            mlat, mlon = self.apx.geo2apex(90 - theta, phi, (r - RE) * 1e-3)
            lat_out, phi_out, _ = self.apx.apex2geo(mlat, mlon, (r_dest - RE) * 1e-3)
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
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))

        if self.kind == "radial":
            raise ValueError("Conjugate coordinates do not exist with radial field lines")

        if is_dipole_kind(self.kind):
            theta_conj, phi_conj = (180 - theta, phi)

        if self.kind == "igrf":
            h = (r - RE) * 1e-3
            mlat, mlon = self.apx.geo2apex(90 - theta, phi, h)
            glat, phi_conj, _ = self.apx.apex2geo(-mlat, mlon, h)
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
        size = r.size
        d1 = np.empty((3, size))
        d2 = np.empty((3, size))
        d3 = np.empty((3, size))
        e1 = np.empty((3, size))
        e2 = np.empty((3, size))
        e3 = np.empty((3, size))

        if self.kind == "radial":
            e = np.vstack((np.ones(size), np.zeros(size), np.zeros(size)))
            n = np.vstack((np.zeros(size), np.ones(size), np.zeros(size)))
            u = np.vstack((np.zeros(size), np.zeros(size), np.ones(size)))
            d1, e1 = e
            d2, e2 = n * np.sign(self.B(RE, 0, 0)[0]) * (-1)
            d3, e3 = u * np.sign(self.B(RE, 0, 0)[0])

        if is_dipole_kind(self.kind):
            _d1, _d2, _d3, _e1, _e2, _e3 = self.dpl.get_apex_base_vectors(
                90 - theta, r * 1e-3, R=RE * 1e-3
            )
            # Transform vectors from east north up to r, theta phi.
            d1[0] = _d1[2]  # Radial
            d2[0] = _d2[2]  # Radial
            d3[0] = _d3[2]  # Radial
            e1[0] = _e1[2]  # Radial
            e2[0] = _e2[2]  # Radial
            e3[0] = _e3[2]  # Radial
            d1[1] = -_d1[1]  # Theta
            d2[1] = -_d2[1]  # Theta
            d3[1] = -_d3[1]  # Theta
            e1[1] = -_e1[1]  # Theta
            e2[1] = -_e2[1]  # Theta
            e3[1] = -_e3[1]  # Theta
            d1[2] = _d1[0]  # Phi
            d2[2] = _d2[0]  # Phi
            d3[2] = _d3[0]  # Phi
            e1[2] = _e1[0]  # Phi
            e2[2] = _e2[0]  # Phi
            e3[2] = _e3[0]  # Phi

        if self.kind == "igrf":
            _, _, _, _, _, _, _d1, _d2, _d3, _e1, _e2, _e3 = self.apx.basevectors_apex(
                90 - theta, phi, (r - RE) * 1e-3, coords="geo"
            )
            # Transform vectors from east north up to r, theta phi.
            d1[0] = _d1[2]  # Radial
            d1[1] = -_d1[1]  # Theta
            d1[2] = _d1[0]  # Phi
            d2[0] = _d2[2]  # Radial
            d2[1] = -_d2[1]  # Theta
            d2[2] = _d2[0]  # Phi
            d3[0] = _d3[2]  # Radial
            d3[1] = -_d3[1]  # Theta
            d3[2] = _d3[0]  # Phi
            e1[0] = _e1[2]  # Radial
            e1[1] = -_e1[1]  # Theta
            e1[2] = _e1[0]  # Phi
            e2[0] = _e2[2]  # Radial
            e2[1] = -_e2[1]  # Theta
            e2[2] = _e2[0]  # Phi
            e3[0] = _e3[2]  # Radial
            e3[1] = -_e3[1]  # Theta
            e3[2] = _e3[0]  # Phi

        return (d1, d2, d3, e1, e2, e3)

    def dip_equator(self, phi, theta=90):
        """Calculate colatitude of given magnetic latitude at phi.

        Parameters
        ----------
        phi : array-like
            Longitude [deg] at which to calculate the dip equator.
        theta : float, optional
            Magnetic latitude.

        Returns
        -------
        array
            The co-latitude of the dip equator at the given longitude.
        """
        phi = np.array(phi) % 360

        if self.kind == "radial":
            print('dip_equator: Not defined for mainfield.kind=="radial"')
            return np.full_like(phi, np.nan)

        if is_dipole_kind(self.kind):
            return np.zeros_like(phi) + theta

        if self.kind == "igrf":
            mlon = np.linspace(0, 360, 360)
            # Calculate latitude of evenly spaced points.
            lat, lon, _ = self.apx.apex2geo(90 - theta, mlon, self.apx.refh)
            # Interpolate to phi.
            return (np.interp(phi.reshape(-1), lon % 360, 90 - lat, period=360)).reshape(phi.shape)
