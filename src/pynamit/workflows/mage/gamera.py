"""Read and remap GAMERA inner-boundary fields for MAGE forcing."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from kompe.constants import EARTH_RADIUS_M
from kompe.math import backend_context
from kompe.mesh import spherical_triangle_solid_angle

from pynamit.coordinates import decimal_year
from pynamit.geomagnetism import MainField
from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm

GAMERA_EARTH_SPEED_SCALE_M_S = 1.0e5
MJD_EPOCH = dt.datetime(1858, 11, 17)


def _datetime_from_mjd(value: float) -> dt.datetime:
    """Convert one finite MJD value to a naive UTC datetime."""
    mjd = float(value)
    if not np.isfinite(mjd):
        raise RuntimeError("Source MJD time must be finite.")
    return MJD_EPOCH + dt.timedelta(days=mjd)


def _kaiju_sm_transform_time(event_time: dt.datetime) -> dt.datetime:
    """Return the whole-second time used by Kaiju's ``mjdRECALC``."""
    if not isinstance(event_time, dt.datetime):
        raise TypeError("Kaiju SM transform time must be a datetime.")
    if event_time.tzinfo is not None:
        event_time = event_time.astimezone(dt.timezone.utc).replace(tzinfo=None)
    # Fortran NINT rounds the non-negative second-of-minute value to the
    # nearest integer, including carry into the next minute.
    return (event_time + dt.timedelta(microseconds=500_000)).replace(microsecond=0)


def _gamera_internal_dipole_axes(mag_m0_nT: float) -> dict[str, np.ndarray]:
    """Return GAMERA dipole-moment and magnetic-pole axes."""
    if not np.isfinite(mag_m0_nT) or mag_m0_nT == 0.0:
        raise ValueError("GAMERA MagM0 must be finite and nonzero.")
    sign = float(np.sign(mag_m0_nT))
    moment_axis = np.array([0.0, 0.0, sign])
    north_axis = -moment_axis
    moment_axis[np.isclose(moment_axis, 0.0)] = 0.0
    north_axis[np.isclose(north_axis, 0.0)] = 0.0
    return {"moment_axis": moment_axis, "north_axis": north_axis}


def _pynamit_dipole_B0_T(mag_m0_nT: float, length_scale_m: float) -> float:
    """Convert GAMERA MagM0 to PynaMIT's reference-radius B0."""
    if not np.isfinite(mag_m0_nT) or mag_m0_nT == 0.0:
        raise ValueError("GAMERA MagM0 must be finite and nonzero.")
    if not np.isfinite(length_scale_m) or length_scale_m <= 0.0:
        raise ValueError("GAMERA length scale must be finite and positive.")
    return abs(float(mag_m0_nT)) * 1e-9 * (float(length_scale_m) / EARTH_RADIUS_M) ** 3


def _centered_dipole_alignment_attrs(event_time: dt.datetime, mag_m0_nT: float) -> dict[str, Any]:
    """Return coordinate alignment for prepared GAMERA forcing."""
    transform_time = _kaiju_sm_transform_time(event_time)
    main_field = MainField(kind="kaiju_dipole", epoch=decimal_year(transform_time))
    alignment = main_field.alignment_metadata(transform_time)
    internal = _gamera_internal_dipole_axes(mag_m0_nT)
    return {
        "gamera_source_coordinate_system": "SM",
        "gamera_internal_magnetic_north_axis": internal["north_axis"],
        "gamera_internal_dipole_moment_axis": internal["moment_axis"],
        **alignment,
    }


def _h5_text(value: Any) -> str:
    """Return one HDF5 text attribute as a stripped string."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    return str(value).strip()


def _gamera_length_scale_m(gsph: Any) -> float:
    """Return the EARTH-normalized GAMERA coordinate scale in metres."""
    with h5py.File(gsph.f0, "r") as file:
        units_id = _h5_text(file.attrs.get("UnitsID", ""))
        if units_id.upper() != "EARTH":
            raise RuntimeError(
                "MAGE preparation requires an EARTH-normalized GAMERA file; "
                f"got UnitsID={units_id!r}."
            )
        if "tScl" not in file.attrs:
            raise RuntimeError("EARTH-normalized GAMERA metadata is missing tScl.")
        time_scale_seconds = float(file.attrs["tScl"])
        if not np.isfinite(time_scale_seconds) or time_scale_seconds <= 0.0:
            raise RuntimeError("GAMERA tScl must be finite and positive.")
        if _h5_text(file.attrs.get("timeID", "")) != "s":
            raise RuntimeError("GAMERA tScl must be expressed in seconds.")
    # Kaiju's EARTH normalization fixes v0=100 km/s and tScl=Rp/v0.
    return time_scale_seconds * GAMERA_EARTH_SPEED_SCALE_M_S


def _gamera_dipole_strength_nT(gsph: Any) -> float:
    """Return GAMERA's required signed dipole strength in nT."""
    with h5py.File(gsph.f0, "r") as file:
        if "MagM0" not in file.attrs:
            raise RuntimeError(
                "GAMERA root metadata is missing the signed dipole strength MagM0. "
                "It is required to align and scale the prepared forcing."
            )
        strength = float(file.attrs["MagM0"])
    if not np.isfinite(strength) or strength == 0.0:
        raise RuntimeError("GAMERA MagM0 must be finite and nonzero.")
    return strength


def _gamera_background_field(
    gsph: Any, inner_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the volume-averaged inner-boundary GAMERA split field."""
    names = ("Bx0", "By0", "Bz0")
    with h5py.File(gsph.f0, "r") as root_file:
        missing = [name for name in names if name not in root_file]
        wrong_units = [
            name
            for name in names
            if name in root_file and _h5_text(root_file[name].attrs.get("Units", "")) != "nT"
        ]
    if missing:
        raise RuntimeError(
            "This preparation path expects Kaiju background-field output. "
            f"Missing root datasets: {missing}. For MAGE/GAMERA Earth simulations, "
            "Kaiju writes total Bx/By/Bz and root Bx0/By0/Bz0."
        )
    if wrong_units:
        raise RuntimeError(f"GAMERA split-background datasets must use nT: {wrong_units}.")
    return tuple(np.asarray(gsph.GetVar(name)[inner_index]) for name in names)


def _validate_gamera_dynamic_field_units(gsph: Any, step: int) -> None:
    """Require physical nT units for saved GAMERA magnetic histories."""
    group_name = f"Step#{step}"
    names = ("Bx", "By", "Bz")
    with h5py.File(gsph.f0, "r") as file:
        if group_name not in file:
            raise RuntimeError(f"GAMERA file is missing {group_name!r}.")
        group = file[group_name]
        missing = [name for name in names if name not in group]
        wrong_units = [
            name
            for name in names
            if name in group and _h5_text(group[name].attrs.get("Units", "")) != "nT"
        ]
    if missing:
        raise RuntimeError(f"GAMERA history {group_name!r} is missing {missing}.")
    if wrong_units:
        raise RuntimeError(f"GAMERA magnetic histories must use nT: {wrong_units}.")


@dataclass(frozen=True)
class _GameraBoundaryGeometry:
    """Geometry of one GAMERA magnetic-field shell."""

    sm_latitude: np.ndarray
    sm_longitude: np.ndarray
    radius_m: np.ndarray
    radial_unit_x: np.ndarray
    radial_unit_y: np.ndarray
    radial_unit_z: np.ndarray
    solid_angle: np.ndarray

    def radial_component(self, bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
        """Return a Cartesian field's radial component on this shell."""
        bx, by, bz = (np.asarray(component) for component in (bx, by, bz))
        if bx.shape != self.radius_m.shape or by.shape != bx.shape or bz.shape != bx.shape:
            raise ValueError("GAMERA boundary-field components must match the boundary geometry.")
        return bx * self.radial_unit_x + by * self.radial_unit_y + bz * self.radial_unit_z


def _gamera_inner_boundary_geometry(
    gsph: Any, inner_index: int, length_scale_m: float
) -> _GameraBoundaryGeometry:
    """Return Kaiju cell centers corresponding to ``B[inner_index]``.

    GAMERA stores ``X/Y/Z`` at cell vertices and magnetic fields at cell
    centers. The selected magnetic shell therefore lies between vertex
    shells ``inner_index`` and ``inner_index + 1``. Kaiju defines the
    location as the volume barycenter of the trilinear cell.
    """
    vertices = _gamera_inner_boundary_vertices(gsph, inner_index)
    x, y, z = np.moveaxis(_trilinear_hexahedron_volume_centers(vertices), -1, 0)

    r_re = np.sqrt(x**2 + y**2 + z**2)
    radial_unit_x = x / r_re
    radial_unit_y = y / r_re
    radial_unit_z = z / r_re
    return _GameraBoundaryGeometry(
        sm_latitude=np.degrees(np.arcsin(np.clip(radial_unit_z, -1.0, 1.0))),
        sm_longitude=np.degrees(np.arctan2(y, x)),
        radius_m=r_re * length_scale_m,
        radial_unit_x=radial_unit_x,
        radial_unit_y=radial_unit_y,
        radial_unit_z=radial_unit_z,
        solid_angle=_gamera_boundary_solid_angle(vertices),
    )


def _gamera_inner_boundary_vertices(gsph: Any, inner_index: int) -> np.ndarray:
    """Return boundary hexahedron vertices in Kaiju's corner order."""
    x, y, z = (
        np.asarray(coordinate[inner_index : inner_index + 2], dtype=float)
        for coordinate in (gsph.X, gsph.Y, gsph.Z)
    )
    positions = np.stack((x, y, z), axis=-1)
    vertices = np.stack(
        (
            positions[0, :-1, :-1],
            positions[1, :-1, :-1],
            positions[1, 1:, :-1],
            positions[0, 1:, :-1],
            positions[0, :-1, 1:],
            positions[1, :-1, 1:],
            positions[1, 1:, 1:],
            positions[0, 1:, 1:],
        ),
        axis=-2,
    )
    if np.any(~np.isfinite(vertices)):
        raise RuntimeError("GAMERA inner-boundary vertices must be finite.")
    return vertices


def _trilinear_hexahedron_volume_centers(vertices: np.ndarray) -> np.ndarray:
    """Return volume barycenters using Kaiju's Gaussian quadrature."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape[-2:] != (8, 3):
        raise ValueError("Hexahedron vertices must have final shape (8, 3).")
    corner_signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    points, weights = np.polynomial.legendre.leggauss(12)
    volume = np.zeros(vertices.shape[:-2], dtype=float)
    first_moment = np.zeros(vertices.shape[:-2] + (3,), dtype=float)
    for i, xi in enumerate(points):
        for j, eta in enumerate(points):
            for k, zeta in enumerate(points):
                factors = (
                    (1.0 + corner_signs[:, 0] * xi)
                    * (1.0 + corner_signs[:, 1] * eta)
                    * (1.0 + corner_signs[:, 2] * zeta)
                    / 8.0
                )
                position = np.einsum("...vc,v->...c", vertices, factors, optimize=True)
                derivatives = (
                    np.stack(
                        (
                            corner_signs[:, 0]
                            * (1.0 + corner_signs[:, 1] * eta)
                            * (1.0 + corner_signs[:, 2] * zeta),
                            (1.0 + corner_signs[:, 0] * xi)
                            * corner_signs[:, 1]
                            * (1.0 + corner_signs[:, 2] * zeta),
                            (1.0 + corner_signs[:, 0] * xi)
                            * (1.0 + corner_signs[:, 1] * eta)
                            * corner_signs[:, 2],
                        ),
                        axis=-1,
                    )
                    / 8.0
                )
                jacobian = np.einsum("...vc,vq->...cq", vertices, derivatives, optimize=True)
                weighted_volume = (
                    weights[i] * weights[j] * weights[k] * np.abs(np.linalg.det(jacobian))
                )
                volume += weighted_volume
                first_moment += weighted_volume[..., None] * position
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise RuntimeError("GAMERA inner-boundary cells must have finite positive volumes.")
    return first_moment / volume[..., None]


def _gamera_boundary_solid_angle(vertices: np.ndarray) -> np.ndarray:
    """Return each boundary cell's solid angle from its vertices."""
    mid_shell = np.stack(
        (
            0.5 * (vertices[..., 0, :] + vertices[..., 1, :]),
            0.5 * (vertices[..., 3, :] + vertices[..., 2, :]),
            0.5 * (vertices[..., 7, :] + vertices[..., 6, :]),
            0.5 * (vertices[..., 4, :] + vertices[..., 5, :]),
        ),
        axis=-2,
    )
    norms = np.linalg.norm(mid_shell, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("GAMERA inner-boundary vertices must have finite nonzero radii.")
    unit = mid_shell / norms
    lower_left, upper_left, upper_right, lower_right = np.moveaxis(unit, -2, 0)
    # Build provider-file geometry on the CPU before input projection.
    with backend_context("numpy"):
        solid_angle = spherical_triangle_solid_angle(
            lower_left, upper_left, upper_right
        ) + spherical_triangle_solid_angle(lower_left, upper_right, lower_right)
    if np.any(~np.isfinite(solid_angle)) or np.any(solid_angle <= 0.0):
        raise RuntimeError("GAMERA inner-boundary cells must have finite positive solid angles.")
    return solid_angle


def _gamera_native_angles(
    sm_latitude: np.ndarray, sm_longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return GAMERA-native colatitude and azimuth in radians.

    GAMERA's spherical grid uses the SM +x axis as its polar axis and
    measures azimuth from +y toward +z.
    """
    sm_latitude, sm_longitude = np.broadcast_arrays(
        np.asarray(sm_latitude, dtype=float), np.asarray(sm_longitude, dtype=float)
    )
    latitude = np.deg2rad(sm_latitude)
    longitude = np.deg2rad(sm_longitude)
    cos_latitude = np.cos(latitude)
    x = cos_latitude * np.cos(longitude)
    y = cos_latitude * np.sin(longitude)
    z = np.sin(latitude)
    colatitude = np.arccos(np.clip(x, -1.0, 1.0))
    azimuth = np.mod(np.arctan2(z, y), 2.0 * np.pi)
    return colatitude, azimuth


class _GameraBoundaryInterpolator:
    """Apply Kaiju-style four-point bilinear interpolation on GAMERA.

    The full inner boundary is a periodic tensor grid in GAMERA's native
    angular coordinates even though it is folded in ordinary SM
    latitude/longitude. Cell transforms are built once and reused for
    every magnetic history. Values at the omitted +x and -x poles are
    reconstructed from the means of their adjacent cell-center rings.
    """

    def __init__(self, source_sm_lat: np.ndarray, source_sm_lon: np.ndarray) -> None:
        source_sm_lat, source_sm_lon = np.broadcast_arrays(
            np.asarray(source_sm_lat, dtype=float), np.asarray(source_sm_lon, dtype=float)
        )
        if source_sm_lat.ndim != 2 or min(source_sm_lat.shape) < 2:
            raise ValueError(
                "A GAMERA boundary grid must be two-dimensional with at least two cells per axis."
            )
        if np.any(~np.isfinite(source_sm_lat)) or np.any(~np.isfinite(source_sm_lon)):
            raise ValueError("GAMERA boundary coordinates must be finite.")

        colatitude, azimuth = _gamera_native_angles(source_sm_lat, source_sm_lon)
        azimuth = np.unwrap(azimuth, axis=1)
        azimuth += 2.0 * np.pi * np.round((azimuth[0, 0] - azimuth[:, [0]]) / (2.0 * np.pi))
        colatitude_axis = np.mean(colatitude, axis=1)
        azimuth_axis = np.mean(azimuth, axis=0)

        self._colatitude_order = np.arange(source_sm_lat.shape[0])
        self._azimuth_order = np.arange(source_sm_lat.shape[1])
        if np.all(np.diff(colatitude_axis) < 0.0):
            self._colatitude_order = self._colatitude_order[::-1]
            colatitude = colatitude[::-1]
            colatitude_axis = colatitude_axis[::-1]
            azimuth = azimuth[::-1]
        if np.all(np.diff(azimuth_axis) < 0.0):
            self._azimuth_order = self._azimuth_order[::-1]
            colatitude = colatitude[:, ::-1]
            azimuth = azimuth[:, ::-1]
            azimuth_axis = azimuth_axis[::-1]
        if np.any(np.diff(colatitude_axis) <= 0.0) or np.any(np.diff(azimuth_axis) <= 0.0):
            raise ValueError("GAMERA native angular coordinates must be monotonic.")
        if colatitude_axis[0] <= 0.0 or colatitude_axis[-1] >= np.pi:
            raise ValueError("GAMERA cell-center colatitudes must lie strictly between its poles.")

        colatitude_step = np.min(np.diff(colatitude_axis))
        azimuth_step = np.min(np.diff(azimuth_axis))
        if np.max(np.abs(colatitude - colatitude_axis[:, None])) >= 0.25 * colatitude_step:
            raise ValueError("GAMERA colatitudes do not form a searchable logical grid.")
        if np.max(np.abs(azimuth - azimuth_axis[None, :])) >= 0.25 * azimuth_step:
            raise ValueError("GAMERA azimuths do not form a searchable logical grid.")

        self._source_shape = source_sm_lat.shape
        self._colatitude_axis = np.concatenate(([0.0], colatitude_axis, [np.pi]))
        self._azimuth_axis = azimuth_axis
        self._periodic_azimuth_axis = np.concatenate(
            (azimuth_axis, [azimuth_axis[0] + 2.0 * np.pi])
        )
        polar_azimuth = np.broadcast_to(azimuth_axis, (1, azimuth_axis.size))
        colatitude = np.vstack(
            (np.zeros_like(polar_azimuth), colatitude, np.full_like(polar_azimuth, np.pi))
        )
        azimuth = np.vstack((polar_azimuth, azimuth, polar_azimuth))
        self._cell_inverse = self._build_cell_inverse(colatitude, azimuth)

    @staticmethod
    def _build_cell_inverse(colatitude: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
        """Return Kaiju's four-corner bilinear transforms."""
        lower_azimuth = azimuth[:-1]
        upper_azimuth = azimuth[1:]
        lower_right_azimuth = np.roll(lower_azimuth, -1, axis=1)
        upper_right_azimuth = np.roll(upper_azimuth, -1, axis=1)
        lower_right_azimuth[:, -1] += 2.0 * np.pi
        upper_right_azimuth[:, -1] += 2.0 * np.pi
        vertex_azimuth = np.stack(
            (lower_azimuth, lower_right_azimuth, upper_azimuth, upper_right_azimuth), axis=-1
        )
        vertex_colatitude = np.stack(
            (
                colatitude[:-1],
                np.roll(colatitude[:-1], -1, axis=1),
                colatitude[1:],
                np.roll(colatitude[1:], -1, axis=1),
            ),
            axis=-1,
        )
        basis = np.stack(
            (
                np.ones_like(vertex_azimuth),
                vertex_azimuth,
                vertex_colatitude,
                vertex_azimuth * vertex_colatitude,
            ),
            axis=-2,
        )
        try:
            inverse = np.linalg.inv(basis)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "GAMERA boundary cells must have invertible angular geometry."
            ) from exc
        if np.any(~np.isfinite(inverse)):
            raise ValueError("GAMERA boundary interpolation geometry must be finite.")
        return inverse

    def interpolate(
        self, values: np.ndarray, *, target_sm_lat: np.ndarray, target_sm_lon: np.ndarray
    ) -> np.ndarray:
        """Interpolate one boundary field at SM target positions."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._source_shape:
            raise ValueError(
                f"GAMERA boundary field shape {values.shape} does not match {self._source_shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("GAMERA boundary interpolation requires finite source values.")
        values = values[np.ix_(self._colatitude_order, self._azimuth_order)]
        values = np.vstack(
            (
                np.full((1, values.shape[1]), np.mean(values[0])),
                values,
                np.full((1, values.shape[1]), np.mean(values[-1])),
            )
        )

        target_sm_lat, target_sm_lon = np.broadcast_arrays(
            np.asarray(target_sm_lat, dtype=float), np.asarray(target_sm_lon, dtype=float)
        )
        target_shape = target_sm_lat.shape
        colatitude, azimuth = _gamera_native_angles(target_sm_lat, target_sm_lon)
        colatitude = colatitude.reshape(-1)
        azimuth = azimuth.reshape(-1)
        if np.any(~np.isfinite(colatitude)) or np.any(~np.isfinite(azimuth)):
            raise ValueError("GAMERA boundary target coordinates must be finite.")
        azimuth = np.mod(azimuth - self._azimuth_axis[0], 2.0 * np.pi) + self._azimuth_axis[0]

        colatitude_index = np.searchsorted(self._colatitude_axis, colatitude, side="right") - 1
        colatitude_index = np.clip(colatitude_index, 0, self._colatitude_axis.size - 2)
        azimuth_index = np.searchsorted(self._periodic_azimuth_axis, azimuth, side="right") - 1
        azimuth_index = np.clip(azimuth_index, 0, self._azimuth_axis.size - 1)
        next_azimuth_index = (azimuth_index + 1) % self._azimuth_axis.size

        target_basis = np.column_stack(
            (np.ones_like(azimuth), azimuth, colatitude, azimuth * colatitude)
        )
        weights = np.einsum(
            "nij,nj->ni", self._cell_inverse[colatitude_index, azimuth_index], target_basis
        )
        # Skewed logical cells can give signed weights; clipping them
        # would break constant and bilinear field reproduction.
        corners = np.column_stack(
            (
                values[colatitude_index, azimuth_index],
                values[colatitude_index, next_azimuth_index],
                values[colatitude_index + 1, azimuth_index],
                values[colatitude_index + 1, next_azimuth_index],
            )
        )
        return np.sum(weights * corners, axis=1).reshape(target_shape)


def _geographic_grid_in_sm(
    latitude: np.ndarray, longitude: np.ndarray, event_time: dt.datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Return Kaiju SM coordinates of a fixed GEO grid."""
    return kaiju_geopack_sm(_kaiju_sm_transform_time(event_time)).geo2sm(latitude, longitude)
