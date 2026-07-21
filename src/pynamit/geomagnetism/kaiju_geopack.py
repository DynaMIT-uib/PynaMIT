"""Kaiju/Geopack centered-dipole and SM-coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import floor
from typing import Any

import dipole
import numpy as np

from pynamit.coordinates import wrap_longitude_180


@dataclass(frozen=True)
class GeopackDipoleCoefficients:
    """Degree-1 internal-field coefficients used by Kaiju Geopack."""

    epoch_value: float
    g10: float
    g11: float
    h11: float

    @property
    def B0_nT(self) -> float:
        """Centered-dipole reference field magnitude in nT."""
        return float(np.sqrt(self.g10**2 + self.g11**2 + self.h11**2))

    @property
    def axis(self) -> np.ndarray:
        """GEO unit vector toward the centered-dipole north pole."""
        axis = -np.array([self.g11, self.h11, self.g10], dtype=float)
        return axis / self.B0_nT


@dataclass(frozen=True)
class KaijuGeopackMAG:
    """Earth-fixed Kaiju/Geopack centered-dipole coordinates."""

    epoch: datetime
    coefficients: GeopackDipoleCoefficients
    geo_to_mag_matrix: np.ndarray

    def __post_init__(self) -> None:
        """Own and validate the GEO-to-MAG rotation matrix."""
        object.__setattr__(
            self,
            "geo_to_mag_matrix",
            _validated_rotation_matrix(self.geo_to_mag_matrix, name="geo_to_mag_matrix"),
        )

    @property
    def mag_to_geo_matrix(self) -> np.ndarray:
        """Return the inverse MAG-to-GEO rotation matrix."""
        return self.geo_to_mag_matrix.T

    def geo2mag(self, lat, lon, east=None, north=None):
        """Convert GEO coordinates and optional vectors to MAG."""
        return _rotate_spherical(self.geo_to_mag_matrix, lat, lon, east=east, north=north)

    def mag2geo(self, lat, lon, east=None, north=None):
        """Convert MAG coordinates and optional vectors to GEO."""
        return _rotate_spherical(self.mag_to_geo_matrix, lat, lon, east=east, north=north)


@dataclass(frozen=True)
class KaijuGeopackSM:
    """Kaiju/Geopack GEO-SM rotation for one epoch."""

    epoch: datetime
    coefficients: GeopackDipoleCoefficients
    geo_to_sm_matrix: np.ndarray

    def __post_init__(self) -> None:
        """Own and validate the GEO-to-SM rotation matrix."""
        object.__setattr__(
            self,
            "geo_to_sm_matrix",
            _validated_rotation_matrix(self.geo_to_sm_matrix, name="geo_to_sm_matrix"),
        )

    @property
    def sm_to_geo_matrix(self) -> np.ndarray:
        """Return the inverse SM-to-GEO rotation matrix."""
        return self.geo_to_sm_matrix.T

    def geo2sm(self, lat, lon, east=None, north=None):
        """Convert GEO coordinates and optional vectors to SM."""
        return _rotate_spherical(self.geo_to_sm_matrix, lat, lon, east=east, north=north)

    def sm2geo(self, lat, lon, east=None, north=None):
        """Convert SM coordinates and optional vectors to GEO."""
        return _rotate_spherical(self.sm_to_geo_matrix, lat, lon, east=east, north=north)


GEOPACK_DIPOLE_COEFFICIENTS: dict[int, tuple[float, float, float]] = {
    # Values are the degree-1 entries from Kaiju src/base/geopack.F90.
    # The full Geopack table contains many higher-degree terms, but
    # SM/CD placement only uses g10, g11, and h11.
    1965: (-30334.0, -2119.0, 5776.0),
    1970: (-30220.0, -2068.0, 5737.0),
    1975: (-30100.0, -2013.0, 5675.0),
    1980: (-29992.0, -1956.0, 5604.0),
    1985: (-29873.0, -1905.0, 5500.0),
    1990: (-29775.0, -1848.0, 5406.0),
    1995: (-29692.0, -1784.0, 5306.0),
    2000: (-29619.4, -1728.2, 5186.1),
    2005: (-29554.6, -1669.0, 5078.0),
    2010: (-29496.57, -1586.42, 4944.26),
    2015: (-29441.46, -1501.77, 4795.99),
    2020: (-29404.8, -1450.9, 4652.5),
}

GEOPACK_2020_SECULAR_VARIATION = (5.7, 7.4, -25.9)


def _validated_rotation_matrix(matrix, *, name: str) -> np.ndarray:
    """Return an immutable proper rotation matrix."""
    matrix = np.array(matrix, dtype=float, copy=True)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be orthogonal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be a proper rotation matrix.")
    matrix.setflags(write=False)
    return matrix


def _datetime_from_decimal_year(epoch: float) -> datetime:
    """Convert a decimal year to a UTC-like datetime."""
    year = int(floor(float(epoch)))
    year_start = datetime(year, 1, 1)
    next_year_start = datetime(year + 1, 1, 1)
    return year_start + timedelta(
        seconds=(float(epoch) - year) * (next_year_start - year_start).total_seconds()
    )


def _as_datetime(epoch: float | datetime) -> datetime:
    """Return ``epoch`` as a naive UTC datetime."""
    if isinstance(epoch, datetime):
        if epoch.tzinfo is not None:
            epoch = epoch.astimezone(timezone.utc).replace(tzinfo=None)
        return epoch
    return _datetime_from_decimal_year(float(epoch))


def _geopack_epoch_value(epoch: float | datetime) -> float:
    """Return the year used by Kaiju Geopack interpolation."""
    if isinstance(epoch, datetime):
        epoch = _as_datetime(epoch)
        return epoch.year + (epoch.timetuple().tm_yday - 1) / 365.25
    return float(epoch)


def axis_lat_lon(axis: np.ndarray) -> np.ndarray:
    """Return geocentric latitude/longitude of a Cartesian unit axis."""
    unit = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("axis must have a finite non-zero norm.")
    unit = unit / norm
    lat = np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0)))
    lon = 0.0
    if not np.isclose(abs(unit[2]), 1.0):
        lon = np.degrees(np.arctan2(unit[1], unit[0]))
    lon = wrap_longitude_180(lon)
    return np.array([lat, lon], dtype=float)


def _spherical_to_cartesian(lat, lon):
    """Return Cartesian unit vectors for lat/lon in degrees."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    return np.vstack((cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)))


def _cartesian_to_spherical(cart):
    """Return latitude/longitude from Cartesian unit vectors."""
    x, y, z = cart
    norm = np.linalg.norm(cart, axis=0)
    x = x / norm
    y = y / norm
    z = z / norm
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = wrap_longitude_180(np.rad2deg(np.arctan2(y, x)))
    return lat, lon


def _east_north_unit_vectors(lat, lon):
    """Return local east and north unit vectors in a spherical frame."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    east = np.vstack((-np.sin(lon_rad), np.cos(lon_rad), np.zeros_like(lon_rad)))
    north = np.vstack(
        (-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad))
    )
    return east, north


def _rotate_spherical(rotation, lat, lon, *, east=None, north=None):
    """Rotate spherical coordinates and optional tangent vectors."""
    if east is None or north is None:
        lat, lon = np.broadcast_arrays(lat, lon)
        shape = lat.shape
        cart_in = _spherical_to_cartesian(lat.ravel(), lon.ravel())
        cart_out = rotation @ cart_in
        lat_out, lon_out = _cartesian_to_spherical(cart_out)
        return lat_out.reshape(shape), lon_out.reshape(shape)

    lat, lon, east, north = np.broadcast_arrays(lat, lon, east, north)
    shape = lat.shape
    flat_lat = lat.ravel()
    flat_lon = lon.ravel()
    flat_east = east.ravel()
    flat_north = north.ravel()

    cart_in = _spherical_to_cartesian(flat_lat, flat_lon)
    cart_out = rotation @ cart_in
    lat_out, lon_out = _cartesian_to_spherical(cart_out)

    east_in, north_in = _east_north_unit_vectors(flat_lat, flat_lon)
    vector_in = east_in * flat_east + north_in * flat_north
    vector_out = rotation @ vector_in
    east_out_basis, north_out_basis = _east_north_unit_vectors(lat_out, lon_out)
    east_out = np.sum(vector_out * east_out_basis, axis=0)
    north_out = np.sum(vector_out * north_out_basis, axis=0)

    return (
        lat_out.reshape(shape),
        lon_out.reshape(shape),
        east_out.reshape(shape),
        north_out.reshape(shape),
    )


def _linear_coefficients(
    epoch_value: float, start_year: int, end_year: int
) -> GeopackDipoleCoefficients:
    """Interpolate one five-year Geopack coefficient interval."""
    f2 = (epoch_value - start_year) / (end_year - start_year)
    f1 = 1.0 - f2
    start = np.array(GEOPACK_DIPOLE_COEFFICIENTS[start_year], dtype=float)
    end = np.array(GEOPACK_DIPOLE_COEFFICIENTS[end_year], dtype=float)
    g10, g11, h11 = f1 * start + f2 * end
    return GeopackDipoleCoefficients(epoch_value, float(g10), float(g11), float(h11))


def _geopack_sun_08(epoch: datetime):
    """Return the SUN_08 quantities needed by Kaiju RECALC_08."""
    rad = 57.295779513
    year = epoch.year
    iday = epoch.timetuple().tm_yday
    seconds = epoch.hour * 3600.0 + epoch.minute * 60.0 + epoch.second + epoch.microsecond * 1.0e-6
    fday = seconds / 86400.0
    dj = 365 * (year - 1900) + (year - 1901) // 4 + iday - 0.5 + fday
    t = dj / 36525.0
    vl = np.mod(279.696678 + 0.9856473354 * dj, 360.0)
    gst = np.mod(279.690983 + 0.9856473354 * dj + 360.0 * fday + 180.0, 360.0) / rad
    g = np.mod(358.475845 + 0.985600267 * dj, 360.0) / rad
    slong = (vl + (1.91946 - 0.004789 * t) * np.sin(g) + 0.020094 * np.sin(2.0 * g)) / rad
    if slong > 2.0 * np.pi:
        slong -= 2.0 * np.pi
    if slong < 0.0:
        slong += 2.0 * np.pi
    obliq = (23.45229 - 0.0130125 * t) / rad
    sob = np.sin(obliq)
    slp = slong - 9.924e-5
    sind = sob * np.sin(slp)
    cosd = np.sqrt(1.0 - sind**2)
    sc = sind / cosd
    sdec = np.arctan(sc)
    srasn = np.pi - np.arctan2(np.cos(obliq) / sob * sc, -np.cos(slp) / cosd)
    return gst, slong, srasn, sdec


def _kaiju_geo_to_sm_matrix(epoch: datetime, coefficients: GeopackDipoleCoefficients):
    """Return the GEO-to-SM rotation matrix from Kaiju RECALC_08."""
    g_10 = -coefficients.g10
    g_11 = coefficients.g11
    h_11 = coefficients.h11

    sq = g_11**2 + h_11**2
    sqq = np.sqrt(sq)
    sqr = np.sqrt(g_10**2 + sq)
    sl0 = -h_11 / sqq
    cl0 = -g_11 / sqq
    st0 = sqq / sqr
    ct0 = g_10 / sqr
    stcl = st0 * cl0
    stsl = st0 * sl0

    gst, _, srasn, sdec = _geopack_sun_08(epoch)
    s1 = np.cos(srasn) * np.cos(sdec)
    s2 = np.sin(srasn) * np.cos(sdec)
    s3 = np.sin(sdec)

    # Kaiju RECALC uses VGSEX=-400, VGSEY=VGSEZ=0 in its wrapper,
    # so GSW=GSM.
    x1, x2, x3 = s1, s2, s3

    cgst = np.cos(gst)
    sgst = np.sin(gst)
    dip1 = stcl * cgst - stsl * sgst
    dip2 = stcl * sgst + stsl * cgst
    dip3 = ct0

    y1 = dip2 * x3 - dip3 * x2
    y2 = dip3 * x1 - dip1 * x3
    y3 = dip1 * x2 - dip2 * x1
    y_norm = np.sqrt(y1 * y1 + y2 * y2 + y3 * y3)
    y1 /= y_norm
    y2 /= y_norm
    y3 /= y_norm

    z1 = x2 * y3 - x3 * y2
    z2 = x3 * y1 - x1 * y3
    z3 = x1 * y2 - x2 * y1

    a = np.array(
        [
            [x1 * cgst + x2 * sgst, -x1 * sgst + x2 * cgst, x3],
            [y1 * cgst + y2 * sgst, -y1 * sgst + y2 * cgst, y3],
            [z1 * cgst + z2 * sgst, -z1 * sgst + z2 * cgst, z3],
        ],
        dtype=float,
    )

    sps = dip1 * x1 + dip2 * x2 + dip3 * x3
    cps = np.sqrt(1.0 - sps**2)
    gsw_to_sm = np.array([[cps, 0.0, -sps], [0.0, 1.0, 0.0], [sps, 0.0, cps]], dtype=float)
    return gsw_to_sm @ a


def _kaiju_geo_to_mag_matrix(coefficients: GeopackDipoleCoefficients) -> np.ndarray:
    """Return Kaiju's Earth-fixed GEO-to-MAG rotation matrix."""
    z_mag = coefficients.axis
    y_mag = np.cross(np.array([0.0, 0.0, 1.0]), z_mag)
    y_mag /= np.linalg.norm(y_mag)
    x_mag = np.cross(y_mag, z_mag)
    return np.vstack((x_mag, y_mag, z_mag))


def kaiju_geopack_coefficients(epoch: float | datetime) -> GeopackDipoleCoefficients:
    """Return Kaiju/Geopack degree-1 dipole coefficients for an epoch.

    Kaiju's ``RECALC_08`` uses the day of year, not the time of
    day, for coefficient interpolation. Passing a ``datetime``
    reproduces that path. Passing a decimal year uses the decimal
    value directly.
    """
    epoch_value = _geopack_epoch_value(epoch)
    if epoch_value < 2000.0:
        if epoch_value < 1965.0:
            raise ValueError(
                "Kaiju Geopack clamps years below 1965; pass an epoch >= 1965 "
                "for an explicit Kaiju-compatible centered dipole."
            )
        start_year = 1965 + 5 * int((epoch_value - 1965.0) // 5.0)
        return _linear_coefficients(epoch_value, start_year, start_year + 5)
    if epoch_value < 2005.0:
        return _linear_coefficients(epoch_value, 2000, 2005)
    if epoch_value < 2010.0:
        return _linear_coefficients(epoch_value, 2005, 2010)
    if epoch_value < 2015.0:
        return _linear_coefficients(epoch_value, 2010, 2015)
    if epoch_value < 2020.0:
        return _linear_coefficients(epoch_value, 2015, 2020)
    # Kaiju accepts every day in calendar year 2025. Decimal-year
    # values therefore remain valid up to, but not including, 2026.
    if epoch_value < 2026.0:
        dt = epoch_value - 2020.0
        base = np.array(GEOPACK_DIPOLE_COEFFICIENTS[2020], dtype=float)
        secular = np.array(GEOPACK_2020_SECULAR_VARIATION, dtype=float)
        g10, g11, h11 = base + dt * secular
        return GeopackDipoleCoefficients(epoch_value, float(g10), float(g11), float(h11))
    raise ValueError(
        "Kaiju Geopack clamps years above 2025; pass an epoch before 2026 for "
        "an explicit Kaiju-compatible centered dipole."
    )


def kaiju_geopack_dipole(epoch: float | datetime, *, B0: float | None = None) -> dipole.Dipole:
    """Return a ``dipole.Dipole`` with Kaiju/Geopack alignment.

    Parameters
    ----------
    epoch
        Decimal year or ``datetime`` for Kaiju/Geopack coefficient
        interpolation.
    B0
        Optional equatorial ground field magnitude in Tesla. If
        omitted, the degree-1 Geopack magnitude is used.
    """
    coefficients = kaiju_geopack_coefficients(epoch)
    north_pole = axis_lat_lon(coefficients.axis)
    B0_nT = coefficients.B0_nT if B0 is None else float(B0) * 1e9
    dpl = dipole.Dipole(dipole_pole=tuple(north_pole), B0=B0_nT)
    dpl.kaiju_geopack_coefficients = coefficients
    dpl.kaiju_geopack_epoch_value = coefficients.epoch_value
    return dpl


def kaiju_geopack_mag(epoch: float | datetime) -> KaijuGeopackMAG:
    """Return the Earth-fixed Kaiju/Geopack GEO-MAG transform."""
    epoch_datetime = _as_datetime(epoch)
    coefficients = kaiju_geopack_coefficients(epoch)
    return KaijuGeopackMAG(
        epoch=epoch_datetime,
        coefficients=coefficients,
        geo_to_mag_matrix=_kaiju_geo_to_mag_matrix(coefficients),
    )


def kaiju_geopack_sm(epoch: float | datetime) -> KaijuGeopackSM:
    """Return the Kaiju/Geopack GEO-SM transform for one epoch."""
    epoch_datetime = _as_datetime(epoch)
    coefficients = kaiju_geopack_coefficients(epoch)
    return KaijuGeopackSM(
        epoch=epoch_datetime,
        coefficients=coefficients,
        geo_to_sm_matrix=_kaiju_geo_to_sm_matrix(epoch_datetime, coefficients),
    )


def kaiju_geopack_alignment(
    epoch: float | datetime, *, magnetic_epoch: float | datetime | None = None
) -> dict[str, Any]:
    """Return internal-MAG and timestamped-SM alignment metadata.

    ``epoch`` is the physical time of the Sun-aligned SM transform.
    ``magnetic_epoch`` optionally fixes the internal Earth-attached MAG
    frame and centered-dipole coefficients at a separate simulation
    epoch.
    """
    magnetic_epoch = epoch if magnetic_epoch is None else magnetic_epoch
    dpl = kaiju_geopack_dipole(magnetic_epoch)
    mag = kaiju_geopack_mag(magnetic_epoch)
    sm = kaiju_geopack_sm(epoch)
    coefficients = dpl.kaiju_geopack_coefficients
    epoch_value = coefficients.epoch_value
    return {
        "dipole_alignment_model": "kaiju_geopack_centered_dipole",
        "dipole_alignment_epoch": epoch_value,
        "dipole_sm_transform_time": sm.epoch.isoformat(),
        "dipole_axis_geo_cartesian": np.asarray(dpl.axis, dtype=float),
        "dipole_north_pole_geo_lat_lon": np.asarray(dpl.north_pole, dtype=float),
        "dipole_south_pole_geo_lat_lon": np.asarray(dpl.south_pole, dtype=float),
        "dipole_geopack_g10_nT": coefficients.g10,
        "dipole_geopack_g11_nT": coefficients.g11,
        "dipole_geopack_h11_nT": coefficients.h11,
        "dipole_geopack_B0_nT": coefficients.B0_nT,
        "dipole_mag_x_axis_geo_cartesian": mag.mag_to_geo_matrix[:, 0],
        "dipole_mag_y_axis_geo_cartesian": mag.mag_to_geo_matrix[:, 1],
        "dipole_mag_z_axis_geo_cartesian": mag.mag_to_geo_matrix[:, 2],
        "dipole_sm_x_axis_geo_cartesian": sm.sm_to_geo_matrix[:, 0],
        "dipole_sm_y_axis_geo_cartesian": sm.sm_to_geo_matrix[:, 1],
        "dipole_sm_z_axis_geo_cartesian": sm.sm_to_geo_matrix[:, 2],
    }


__all__ = [
    "GEOPACK_DIPOLE_COEFFICIENTS",
    "GEOPACK_2020_SECULAR_VARIATION",
    "GeopackDipoleCoefficients",
    "KaijuGeopackMAG",
    "KaijuGeopackSM",
    "axis_lat_lon",
    "kaiju_geopack_alignment",
    "kaiju_geopack_coefficients",
    "kaiju_geopack_dipole",
    "kaiju_geopack_mag",
    "kaiju_geopack_sm",
]
