"""Prepare MAGE/GAMERA/TIEGCM forcing for ``mage_forcing_final.py``.

The expensive TIEGCM height integration is done here once.  The output
HDF5 contains the fields used by the final simulation script:

- ``SP`` and ``SH``: Pedersen and Hall conductance in S.
- ``We``/``Wn``: Pedersen-weighted eastward/northward wind in m/s.
- ``WeH``/``WnH``: Hall-weighted eastward/northward neutral wind in m/s.
- MAGE/REMIX FAC, conductance diagnostics, and inner-boundary magnetic
  field.

The wind integration intentionally stores conductivity-weighted winds,
not a height-resolved ``u x B`` source.  The final forcing script uses
the PynaMIT sheet-radius main field and sheet resistance, matching the
thin-sheet ``JS -> E_S`` closure.

Typical use on the MAGE machine:

    python scripts/simulation/mage_prepare_forcing.py

Edit ``SETTINGS`` below to change paths or run parameters. By default,
the GAMERA directory is ``/disk/Gamera_Dong``. Output is written under
``scripts/simulation/mage_prepared``.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.interpolate import griddata

from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism import MainField, decimal_year
from pynamit.simulation.workflows.mage import (
    centered_dipole_alignment_attrs,
    gamera_internal_dipole_axes,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GAMERA_DIR = Path("/disk/Gamera_Dong")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "mage_prepared"
DEFAULT_OUTPUT_NAME = "mage_prepared_forcing.h5"
DEFAULT_TAG = "msphere"

FALLBACK_EARTH_RADIUS_M = 6371.0e3
FILL_THRESHOLD = 1e30


@dataclass(frozen=True)
class MagePrepareSettings:
    """Defaults intended to be edited for preparation runs."""

    gamera_dir: Path = DEFAULT_GAMERA_DIR
    gamera_subdir: str = "gamera"
    tag: str = DEFAULT_TAG
    inner_index: int = 0
    tiegcm_nc: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_name: str = DEFAULT_OUTPUT_NAME
    conductance_source: str = "computed"
    compression: str = "lzf"
    max_steps: int | None = None


SETTINGS = MagePrepareSettings()


def resolve_tiegcm_path(gamera_dir: Path, explicit_path: str | Path | None) -> Path:
    """Resolve the TIEGCM NetCDF path."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"TIEGCM NetCDF does not exist: {path}")
        return path

    matches = sorted(gamera_dir.glob("*sech_tie*.nc"))
    if not matches:
        raise FileNotFoundError(f"Could not find a '*sech_tie*.nc' TIEGCM file in {gamera_dir}")
    if len(matches) > 1:
        print(f"Found multiple TIEGCM files; using {matches[0]}", flush=True)
    return matches[0]


def resolve_gamera_run_dir(gamera_dir: Path, gamera_subdir: str, tag: str) -> Path:
    """Resolve the directory that contains GAMERA/REMIX files."""
    direct_mix = gamera_dir / f"{tag}.mix.h5"
    direct_serial = gamera_dir / f"{tag}.gam.h5"
    direct_mpi = sorted(gamera_dir.glob(f"{tag}_*.gam.h5"))
    if direct_mix.exists() or direct_serial.exists() or direct_mpi:
        return gamera_dir

    nested = gamera_dir / gamera_subdir
    nested_mix = nested / f"{tag}.mix.h5"
    nested_serial = nested / f"{tag}.gam.h5"
    nested_mpi = sorted(nested.glob(f"{tag}_*.gam.h5"))
    if nested_mix.exists() or nested_serial.exists() or nested_mpi:
        return nested

    raise FileNotFoundError(
        f"Could not find GAMERA/REMIX files for tag {tag!r} in {gamera_dir} or {nested}"
    )


def gamera_length_scale_m(gsph: Any) -> float:
    """Return the GAMERA-to-meter length scale."""
    try:
        import h5py

        with h5py.File(gsph.f0, "r") as file:
            units_id = file.attrs.get("UnitsID", b"")
            if isinstance(units_id, bytes):
                units_id = units_id.decode("ascii", errors="ignore")
            if str(units_id).upper().startswith("EARTH") and "tScl" in file.attrs:
                return float(file.attrs["tScl"]) * 1.0e5
    except Exception:
        pass

    return FALLBACK_EARTH_RADIUS_M


def gamera_magnetic_moment_nT(gsph: Any) -> float | None:
    """Return the signed GAMERA dipole moment in nT if available."""
    try:
        with h5py.File(gsph.f0, "r") as file:
            if "MagM0" in file.attrs:
                return float(file.attrs["MagM0"])
    except Exception:
        pass
    return None


def read_nc_step(dataset: Any, name: str, step: int) -> np.ndarray:
    """Read a NetCDF variable time slice while suppressing warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="WARNING: missing_value not used since it.*", category=UserWarning
        )
        return np.asarray(dataset.variables[name][step])


def replace_fill(values: np.ndarray) -> np.ndarray:
    """Replace TIEGCM fill values with NaN."""
    array = np.asarray(values, dtype=float)
    array[array > FILL_THRESHOLD] = np.nan
    return array


def weighted_wind(
    numerator_sigma: np.ndarray,
    denominator_conductance: np.ndarray,
    wind_east: np.ndarray,
    wind_north: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conductivity-weighted eastward and northward winds."""
    east_num = np.nansum(numerator_sigma * wind_east * dz, axis=0)
    north_num = np.nansum(numerator_sigma * wind_north * dz, axis=0)
    east = np.divide(
        east_num,
        denominator_conductance,
        out=np.zeros_like(east_num),
        where=denominator_conductance > 0.0,
    )
    north = np.divide(
        north_num,
        denominator_conductance,
        out=np.zeros_like(north_num),
        where=denominator_conductance > 0.0,
    )
    return east.astype(np.float32), north.astype(np.float32)


def integrate_tiegcm_step(
    dataset: Any, step: int, conductance_source: str
) -> dict[str, np.ndarray]:
    """Height-integrate conductivities and weighted winds."""
    sigma_p = replace_fill(read_nc_step(dataset, "SIGMA_PED", step))
    sigma_h = replace_fill(read_nc_step(dataset, "SIGMA_HAL", step))
    height_m = replace_fill(read_nc_step(dataset, "ZG", step)) / 100.0
    wind_east = replace_fill(read_nc_step(dataset, "UN", step))[:-1] * 1e-2
    wind_north = replace_fill(read_nc_step(dataset, "VN", step))[:-1] * 1e-2

    dz = np.diff(height_m, axis=0)
    sigma_p_layer = sigma_p[:-1]
    sigma_h_layer = sigma_h[:-1]
    sigma_p_int = np.nansum(sigma_p_layer * dz, axis=0)
    sigma_h_int = np.nansum(sigma_h_layer * dz, axis=0)

    if conductance_source == "native":
        sigma_p_out = replace_fill(read_nc_step(dataset, "gzigm1", step))
        sigma_h_out = replace_fill(read_nc_step(dataset, "gzigm2", step))
    else:
        sigma_p_out = sigma_p_int
        sigma_h_out = sigma_h_int

    u_p_east, u_p_north = weighted_wind(sigma_p_layer, sigma_p_out, wind_east, wind_north, dz)
    u_h_east, u_h_north = weighted_wind(sigma_h_layer, sigma_h_out, wind_east, wind_north, dz)

    return {
        "SP": sigma_p_out.astype(np.float32),
        "SH": sigma_h_out.astype(np.float32),
        "We": u_p_east,
        "Wn": u_p_north,
        "WeH": u_h_east,
        "WnH": u_h_north,
    }


def centered_inner_boundary_grid(
    gsph: Any, inner_index: int, length_scale_m: float
) -> tuple[np.ndarray, ...]:
    """Return centered inner-boundary grid and helper arrays."""
    x = gsph.X[inner_index]
    y = gsph.Y[inner_index]
    z = gsph.Z[inner_index]

    x = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    y = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])
    z = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])

    r_re = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r_re)
    phi = np.arctan2(y, x)

    glat = 90.0 - np.degrees(theta)
    glon = np.degrees(phi)
    r_m = r_re * length_scale_m

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    return glat, glon, r_m, sin_theta, cos_theta, sin_phi, cos_phi


def spherical_components(
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    sin_phi: np.ndarray,
    cos_phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian components to spherical Br, Btheta, Bphi."""
    br = bx * sin_theta * cos_phi + by * sin_theta * sin_phi + bz * cos_theta
    btheta = bx * cos_theta * cos_phi + by * cos_theta * sin_phi - bz * sin_theta
    bphi = -bx * sin_phi + by * cos_phi
    return br, btheta, bphi


def interpolate_to_tiegcm_grid(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """Interpolate a REMIX field onto the TIEGCM geographic grid."""
    return griddata(
        (source_lon.reshape(-1), source_lat.reshape(-1)),
        np.asarray(values).reshape(-1),
        (target_lon, target_lat),
        method="linear",
    )


def merge_south_with_north(south: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Fill NaNs in southern interpolation with northern values."""
    output = np.array(south, copy=True)
    mask = np.isnan(output)
    output[mask] = north[mask]
    return output


def remix_hemisphere_fields(
    ion: Any,
    hemisphere: str,
    coordinate_field: MainField,
    mlat: np.ndarray,
    sm_lon: np.ndarray,
    tiegcm_lon: np.ndarray,
    tiegcm_lat: np.ndarray,
    event_time: dt.datetime,
) -> dict[str, np.ndarray]:
    """Return one REMIX hemisphere interpolated onto the TIEGCM grid."""
    ion.init_vars(hemisphere)
    sign = -1.0 if hemisphere == "SOUTH" else 1.0
    lat_mag = sign * mlat

    sigma_h = ion.variables["sigmah"]["data"]
    sigma_p = ion.variables["sigmap"]["data"]
    fac = ion.variables["current"]["data"]
    currents = ion.hCurrents()
    jh_north, jh_east = -currents[6], currents[7]
    jp_north, jp_east = -currents[8], currents[9]

    scalar_lat, scalar_lon = coordinate_field.model_to_geo_coordinates(
        lat_mag, sm_lon, event_time=event_time
    )
    _, _, jh_east, jh_north = coordinate_field.model_to_geo_coordinates(
        lat_mag, sm_lon, jh_east, jh_north, event_time=event_time
    )
    _, _, jp_east, jp_north = coordinate_field.model_to_geo_coordinates(
        lat_mag, sm_lon, jp_east, jp_north, event_time=event_time
    )

    return {
        "SH_G": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, sigma_h, tiegcm_lon, tiegcm_lat
        ),
        "SP_G": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, sigma_p, tiegcm_lon, tiegcm_lat
        ),
        "FAC": interpolate_to_tiegcm_grid(scalar_lat, scalar_lon, fac, tiegcm_lon, tiegcm_lat),
        "JHe": interpolate_to_tiegcm_grid(scalar_lat, scalar_lon, jh_east, tiegcm_lon, tiegcm_lat),
        "JHn": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, jh_north, tiegcm_lon, tiegcm_lat
        ),
        "JPe": interpolate_to_tiegcm_grid(scalar_lat, scalar_lon, jp_east, tiegcm_lon, tiegcm_lat),
        "JPn": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, jp_north, tiegcm_lon, tiegcm_lat
        ),
    }


def remix_fields_for_step(
    remix_file: Path,
    step: int,
    event_time: dt.datetime,
    tiegcm_lon: np.ndarray,
    tiegcm_lat: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read one REMIX step and interpolate north/south fields."""
    try:
        import kaipy.remix.remix as remix
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mage_prepare_forcing.py needs kaipy.remix to read REMIX files. "
            f"Missing module: {exc.name!r}. Run it in the MAGE/GAMERA environment "
            "where kaipy and its dependencies are installed."
        ) from exc

    coordinate_field = MainField(kind="kaiju_dipole", epoch=decimal_year(event_time))
    ion = remix.remix(str(remix_file), step)
    _, _, theta, phi = ion.cartesianCellCenters()
    mlat = 90.0 - theta / np.pi * 180.0
    sm_lon = wrap_longitude_180(phi / np.pi * 180.0)

    north = remix_hemisphere_fields(
        ion, "NORTH", coordinate_field, mlat, sm_lon, tiegcm_lon, tiegcm_lat, event_time
    )
    south = remix_hemisphere_fields(
        ion, "SOUTH", coordinate_field, mlat, sm_lon, tiegcm_lon, tiegcm_lat, event_time
    )
    return {key: merge_south_with_north(south[key], north[key]) for key in south}


def h5_dataset_kwargs(compression: str) -> dict[str, Any]:
    """Return h5py dataset creation options."""
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def create_output_datasets(
    output: h5py.File,
    n_steps: int,
    ion_shape: tuple[int, int],
    inner_shape: tuple[int, int],
    compression: str,
) -> None:
    """Create all time-dependent output datasets."""
    kwargs = h5_dataset_kwargs(compression)
    for name in ("SH", "SP", "We", "Wn", "WeH", "WnH"):
        output.create_dataset(name, shape=(n_steps, *ion_shape), dtype="f4", **kwargs)
    for name in ("FAC", "JHe", "JHn", "JPe", "JPn", "SH_G", "SP_G"):
        output.create_dataset(name, shape=(n_steps, *ion_shape), dtype="f4", **kwargs)
    for name in ("Be", "Bn", "Bu", "Be0", "Bn0", "Bu0"):
        output.create_dataset(name, shape=(n_steps, *inner_shape), dtype="f4", **kwargs)


def write_static_datasets(
    output: h5py.File,
    time_values: np.ndarray,
    event_time: dt.datetime,
    tiegcm_lat: np.ndarray,
    tiegcm_lon: np.ndarray,
    inner_lat: np.ndarray,
    inner_lon: np.ndarray,
    inner_r: np.ndarray,
    settings: MagePrepareSettings,
    gamera_run_dir: Path,
    length_scale_m: float,
    mag_m0_nT: float | None,
    tiegcm_path: Path,
) -> None:
    """Write static datasets and metadata."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    output.create_dataset(
        "time", data=np.asarray(time_values, dtype=string_dtype), dtype=string_dtype
    )
    output.create_dataset("glat", data=tiegcm_lat)
    output.create_dataset("glon", data=tiegcm_lon)
    output.create_dataset("Blat", data=inner_lat)
    output.create_dataset("Blon", data=inner_lon)
    output.create_dataset("r", data=inner_r)
    output.attrs["gamera_dir"] = str(Path(settings.gamera_dir).expanduser())
    output.attrs["gamera_run_dir"] = str(gamera_run_dir)
    output.attrs["tiegcm_nc"] = str(tiegcm_path)
    output.attrs["conductance_source"] = settings.conductance_source
    output.attrs["wind_weighting"] = (
        "Pedersen datasets We/Wn; Hall datasets WeH/WnH; final forcing uses "
        "sheet-radius B and b for the electrodynamic source"
    )
    output.attrs["remix_tag"] = settings.tag
    output.attrs["FAC_convention"] = "upward_positive_kaipy_remix_init_vars"
    output.attrs["gamera_inner_index"] = int(settings.inner_index)
    output.attrs["gamera_length_scale_m"] = float(length_scale_m)
    output.attrs["gamera_B_output"] = "Kaiju Bx/By/Bz total field, with B0 active"
    output.attrs["prepared_B_output"] = (
        "Bu/Bn/Be are perturbations from Bx/By/Bz minus Bx0/By0/Bz0"
    )
    for name, value in centered_dipole_alignment_attrs(event_time, mag_m0_nT).items():
        output.attrs[name] = value
    if mag_m0_nT is not None:
        output.attrs["gamera_mag_m0_nT"] = float(mag_m0_nT)
        output.attrs["gamera_dipole_B0_T"] = abs(float(mag_m0_nT)) * 1e-9
    output.attrs["RM"] = float(np.nanmean(inner_r))
    output.attrs["RM_min"] = float(np.nanmin(inner_r))
    output.attrs["RM_max"] = float(np.nanmax(inner_r))


def validate_settings(settings: MagePrepareSettings) -> None:
    """Validate in-script preparation settings."""
    if settings.conductance_source not in ("computed", "native"):
        raise ValueError(
            "conductance_source must be 'computed' or 'native'; "
            f"got {settings.conductance_source!r}."
        )
    if settings.compression not in ("lzf", "gzip", "none"):
        raise ValueError(
            f"compression must be 'lzf', 'gzip', or 'none'; got {settings.compression!r}."
        )
    if settings.inner_index < 0:
        raise ValueError(f"inner_index must be non-negative; got {settings.inner_index}.")


def prepare_forcing(settings: MagePrepareSettings = SETTINGS) -> Path:
    """Prepare the HDF5 forcing file."""
    validate_settings(settings)
    from netCDF4 import Dataset

    try:
        import kaipy.gamera.magsphere as msph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mage_prepare_forcing.py needs kaipy to read GAMERA/REMIX files. "
            f"Missing module: {exc.name!r}. Run it in the MAGE/GAMERA environment "
            "where kaipy and its dependencies are installed."
        ) from exc

    gamera_dir = Path(settings.gamera_dir).expanduser()
    tiegcm_path = resolve_tiegcm_path(gamera_dir, settings.tiegcm_nc)
    gamera_run_dir = resolve_gamera_run_dir(gamera_dir, settings.gamera_subdir, settings.tag)
    remix_file = gamera_run_dir / f"{settings.tag}.mix.h5"
    if not remix_file.exists():
        raise FileNotFoundError(f"REMIX file does not exist: {remix_file}")
    output_dir = Path(settings.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.output_name

    print(f"Using GAMERA directory: {gamera_dir}", flush=True)
    print(f"Using TIEGCM file: {tiegcm_path}", flush=True)
    print(f"Using REMIX file: {remix_file}", flush=True)
    print(f"Writing prepared forcing: {output_path}", flush=True)

    gsph = msph.GamsphPipe(str(gamera_run_dir), settings.tag, doFast=False)
    if settings.inner_index >= gsph.X.shape[0] - 1:
        raise ValueError(
            f"inner_index must be between 0 and {gsph.X.shape[0] - 2}; got {settings.inner_index}."
        )
    length_scale_m = gamera_length_scale_m(gsph)
    mag_m0_nT = gamera_magnetic_moment_nT(gsph)
    with h5py.File(gsph.f0, "r") as root_file:
        missing_background = [name for name in ("Bx0", "By0", "Bz0") if name not in root_file]
        if missing_background:
            raise RuntimeError(
                "This preparation path expects Kaiju background-field output. "
                f"Missing root datasets: {missing_background}. "
                "For MAGE/GAMERA Earth runs, Kaiju writes total Bx/By/Bz and "
                "root Bx0/By0/Bz0, and the prepared Br is total minus B0."
            )
    print(f"Using GAMERA length scale: {length_scale_m:.6g} m", flush=True)
    if mag_m0_nT is not None:
        axes = gamera_internal_dipole_axes(mag_m0_nT)
        print(f"Using GAMERA dipole MagM0: {mag_m0_nT:.6g} nT", flush=True)
        print(
            "GAMERA internal moment axis: "
            f"{axes['moment_axis'][0]:.3g}, {axes['moment_axis'][1]:.3g}, "
            f"{axes['moment_axis'][2]:.3g}; magnetic north axis: "
            f"{axes['north_axis'][0]:.3g}, {axes['north_axis'][1]:.3g}, "
            f"{axes['north_axis'][2]:.3g}",
            flush=True,
        )
    print(f"Using GAMERA inner index: {settings.inner_index}", flush=True)
    n_available = len(gsph.UT) - 1
    if settings.max_steps is not None:
        n_available = min(n_available, int(settings.max_steps))

    with Dataset(tiegcm_path, mode="r") as tiegcm:
        n_steps = min(n_available, tiegcm.variables["gzigm1"].shape[0])
        lon = np.asarray(tiegcm.variables["lon"][:], dtype=float)
        lon[lon < 0.0] += 360.0
        lat = np.asarray(tiegcm.variables["lat"][:], dtype=float)
        tiegcm_lon, tiegcm_lat = np.meshgrid(lon, lat)

        time_values = np.array(
            [value.replace(microsecond=0).isoformat() for value in gsph.UT[1 : n_steps + 1]],
            dtype="S19",
        )

        inner_lat, inner_lon, inner_r, sin_theta, cos_theta, sin_phi, cos_phi = (
            centered_inner_boundary_grid(gsph, settings.inner_index, length_scale_m)
        )
        # Kaiju gioH5 writes Bx/By/Bz as total field when
        # Model%doBackground is true, and root Bx0/By0/Bz0 as Gr%B0.
        # PynaMIT needs the perturbation.
        bx0 = gsph.GetVar("Bx0")[settings.inner_index]
        by0 = gsph.GetVar("By0")[settings.inner_index]
        bz0 = gsph.GetVar("Bz0")[settings.inner_index]
        br0, btheta0, bphi0 = spherical_components(
            bx0, by0, bz0, sin_theta, cos_theta, sin_phi, cos_phi
        )

        with h5py.File(output_path, "w") as output:
            write_static_datasets(
                output,
                time_values,
                gsph.UT[1],
                tiegcm_lat,
                tiegcm_lon,
                inner_lat,
                inner_lon,
                inner_r,
                settings,
                gamera_run_dir,
                length_scale_m,
                mag_m0_nT,
                tiegcm_path,
            )
            create_output_datasets(
                output, n_steps, tiegcm_lat.shape, inner_lat.shape, settings.compression
            )

            for out_step in range(n_steps):
                gamera_step = gsph.s0 + out_step + 1
                event_time = gsph.UT[out_step + 1]
                print(
                    f"Preparing step {out_step + 1} of {n_steps}: {event_time.isoformat()}",
                    flush=True,
                )

                integrated = integrate_tiegcm_step(tiegcm, out_step, settings.conductance_source)
                for key, values in integrated.items():
                    output[key][out_step] = values

                remix_values = remix_fields_for_step(
                    remix_file, gamera_step, event_time, tiegcm_lon, tiegcm_lat
                )
                for key, values in remix_values.items():
                    output[key][out_step] = values.astype(np.float32)

                bx = gsph.GetVar("Bx", gamera_step)[settings.inner_index] - bx0
                by = gsph.GetVar("By", gamera_step)[settings.inner_index] - by0
                bz = gsph.GetVar("Bz", gamera_step)[settings.inner_index] - bz0
                br, btheta, bphi = spherical_components(
                    bx, by, bz, sin_theta, cos_theta, sin_phi, cos_phi
                )
                output["Be"][out_step] = bphi.astype(np.float32)
                output["Bn"][out_step] = (-btheta).astype(np.float32)
                output["Bu"][out_step] = br.astype(np.float32)
                output["Be0"][out_step] = bphi0.astype(np.float32)
                output["Bn0"][out_step] = (-btheta0).astype(np.float32)
                output["Bu0"][out_step] = br0.astype(np.float32)

    return output_path


def main(settings: MagePrepareSettings = SETTINGS) -> None:
    """Prepare forcing from in-script settings."""
    output_path = prepare_forcing(settings)
    print(f"Prepared forcing written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
