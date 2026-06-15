"""Prepare MAGE/GAMERA/TIEGCM forcing for ``mage_forcing_final.py``.

The expensive TIEGCM height integration is done here once.  The output HDF5
contains the fields used by the final simulation script:

- ``SP`` and ``SH``: Pedersen and Hall conductance in S.
- ``We``/``Wn``: Pedersen-weighted eastward/northward neutral wind in m/s.
- ``WeH``/``WnH``: Hall-weighted eastward/northward neutral wind in m/s.
- MAGE/REMIX FAC, conductance diagnostics, and inner-boundary magnetic field.

Typical use on the MAGE machine:

    python scripts/simulation/mage_prepare_forcing.py --gamera-dir /disk/Gamera_Dong

By default, output is written under ``scripts/simulation/mage_prepared``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from pathlib import Path
from typing import Any

import dipole
import h5py
import numpy as np
from scipy.interpolate import griddata


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GAMERA_DIR = Path("/disk/Gamera_Dong")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "mage_prepared"
DEFAULT_OUTPUT_NAME = "data_H_int_qeff.h5"
DEFAULT_TAG = "msphere"

RE_KM = 6371.0
FILL_THRESHOLD = 1e30


def datetime_to_decimal_year(value: dt.datetime) -> float:
    """Convert datetime to decimal year for dipole transforms."""
    year_start = dt.datetime(value.year, 1, 1)
    next_year_start = dt.datetime(value.year + 1, 1, 1)
    year_seconds = (next_year_start - year_start).total_seconds()
    elapsed = (value - year_start).total_seconds()
    return value.year + elapsed / year_seconds


def resolve_tiegcm_path(gamera_dir: Path, explicit_path: str | None) -> Path:
    """Resolve the TIEGCM NetCDF path."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"TIEGCM NetCDF does not exist: {path}")
        return path

    matches = sorted(gamera_dir.glob("*sech_tie*.nc"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find a '*sech_tie*.nc' TIEGCM file in {gamera_dir}"
        )
    if len(matches) > 1:
        print(f"Found multiple TIEGCM files; using {matches[0]}", flush=True)
    return matches[0]


def output_path_from_args(output_dir: str, output_name: str) -> Path:
    """Return output HDF5 path."""
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / output_name


def read_nc_step(dataset: Any, name: str, step: int) -> np.ndarray:
    """Read a NetCDF variable time slice while suppressing known warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="WARNING: missing_value not used since it.*",
            category=UserWarning,
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
    dataset: Any,
    step: int,
    conductance_source: str,
) -> dict[str, np.ndarray]:
    """Height-integrate TIEGCM conductivities and weighted winds for one step."""
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

    u_p_east, u_p_north = weighted_wind(
        sigma_p_layer,
        sigma_p_out,
        wind_east,
        wind_north,
        dz,
    )
    u_h_east, u_h_north = weighted_wind(
        sigma_h_layer,
        sigma_h_out,
        wind_east,
        wind_north,
        dz,
    )

    return {
        "SP": sigma_p_out.astype(np.float32),
        "SH": sigma_h_out.astype(np.float32),
        "We": u_p_east,
        "Wn": u_p_north,
        "WeH": u_h_east,
        "WnH": u_h_north,
    }


def centered_inner_boundary_grid(gsph: Any) -> tuple[np.ndarray, ...]:
    """Return centered inner-boundary grid and spherical helper arrays."""
    x = gsph.X[0]
    y = gsph.Y[0]
    z = gsph.Z[0]

    x = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    y = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])
    z = 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])

    r_re = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r_re)
    phi = np.arctan2(y, x)

    glat = 90.0 - np.degrees(theta)
    glon = np.degrees(phi)
    r_m = r_re * RE_KM * 1e3

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


def remix_to_geographic(
    dpl: Any,
    mlat: np.ndarray,
    mlon: np.ndarray,
    east: np.ndarray | None = None,
    north: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Convert REMIX magnetic coordinates and optional vector components."""
    if east is None or north is None:
        lat, lon = dpl.mag2geo(mlat, mlon)
        return np.asarray(lat), np.asarray(lon)
    lat, lon, east_geo, north_geo = dpl.mag2geo(mlat, mlon, east, north)
    return np.asarray(lat), np.asarray(lon), np.asarray(east_geo), np.asarray(north_geo)


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
    """Fill NaNs in the southern-grid interpolation with northern values."""
    output = np.array(south, copy=True)
    mask = np.isnan(output)
    output[mask] = north[mask]
    return output


def remix_hemisphere_fields(
    ion: Any,
    hemisphere: str,
    dpl: Any,
    mlat: np.ndarray,
    mlon: np.ndarray,
    tiegcm_lon: np.ndarray,
    tiegcm_lat: np.ndarray,
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

    scalar_lat, scalar_lon = remix_to_geographic(dpl, lat_mag, mlon)
    _, _, jh_east, jh_north = remix_to_geographic(
        dpl, lat_mag, mlon, jh_east, jh_north
    )
    _, _, jp_east, jp_north = remix_to_geographic(
        dpl, lat_mag, mlon, jp_east, jp_north
    )

    return {
        "SH_G": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, sigma_h, tiegcm_lon, tiegcm_lat
        ),
        "SP_G": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, sigma_p, tiegcm_lon, tiegcm_lat
        ),
        "FAC": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, fac, tiegcm_lon, tiegcm_lat
        ),
        "JHe": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, jh_east, tiegcm_lon, tiegcm_lat
        ),
        "JHn": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, jh_north, tiegcm_lon, tiegcm_lat
        ),
        "JPe": interpolate_to_tiegcm_grid(
            scalar_lat, scalar_lon, jp_east, tiegcm_lon, tiegcm_lat
        ),
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
            "Run it in the MAGE/GAMERA environment where kaipy is installed."
        ) from exc

    dpl = dipole.Dipole(datetime_to_decimal_year(event_time))
    ion = remix.remix(str(remix_file), step)
    _, _, theta, phi = ion.cartesianCellCenters()
    mlat = 90.0 - theta / np.pi * 180.0
    local_time_lon = phi / np.pi * 180.0
    mlt = (local_time_lon / 15.0 + 12.0) % 24.0
    mlon = dpl.mlt2mlon(mlt, event_time)

    north = remix_hemisphere_fields(
        ion, "NORTH", dpl, mlat, mlon, tiegcm_lon, tiegcm_lat
    )
    south = remix_hemisphere_fields(
        ion, "SOUTH", dpl, mlat, mlon, tiegcm_lon, tiegcm_lat
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
    tiegcm_lat: np.ndarray,
    tiegcm_lon: np.ndarray,
    inner_lat: np.ndarray,
    inner_lon: np.ndarray,
    inner_r: np.ndarray,
    args: argparse.Namespace,
    tiegcm_path: Path,
) -> None:
    """Write static datasets and metadata."""
    output.create_dataset("time", data=time_values.astype("S19"))
    output.create_dataset("glat", data=tiegcm_lat)
    output.create_dataset("glon", data=tiegcm_lon)
    output.create_dataset("Blat", data=inner_lat)
    output.create_dataset("Blon", data=inner_lon)
    output.create_dataset("r", data=inner_r)
    output.attrs["gamera_dir"] = str(Path(args.gamera_dir).expanduser())
    output.attrs["tiegcm_nc"] = str(tiegcm_path)
    output.attrs["conductance_source"] = args.conductance_source
    output.attrs["wind_weighting"] = "Pedersen datasets We/Wn; Hall datasets WeH/WnH"
    output.attrs["remix_tag"] = args.tag


def prepare_forcing(args: argparse.Namespace) -> Path:
    """Prepare the HDF5 forcing file."""
    from netCDF4 import Dataset
    try:
        import kaipy.gamera.magsphere as msph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mage_prepare_forcing.py needs kaipy to read GAMERA/REMIX files. "
            "Run it in the MAGE/GAMERA environment where kaipy is installed."
        ) from exc

    gamera_dir = Path(args.gamera_dir).expanduser()
    tiegcm_path = resolve_tiegcm_path(gamera_dir, args.tiegcm_nc)
    gamera_run_dir = gamera_dir #/ args.gamera_subdir
    remix_file = gamera_run_dir / f"{args.tag}.mix.h5"
    output_path = output_path_from_args(args.output_dir, args.output_name)

    print(f"Using GAMERA directory: {gamera_dir}", flush=True)
    print(f"Using TIEGCM file: {tiegcm_path}", flush=True)
    print(f"Using REMIX file: {remix_file}", flush=True)
    print(f"Writing prepared forcing: {output_path}", flush=True)

    gsph = msph.GamsphPipe(str(gamera_run_dir), args.tag, doFast=False)
    n_available = len(gsph.UT) - 1
    if args.max_steps is not None:
        n_available = min(n_available, int(args.max_steps))

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
            centered_inner_boundary_grid(gsph)
        )
        bx0 = gsph.GetVar("Bx0")[0]
        by0 = gsph.GetVar("By0")[0]
        bz0 = gsph.GetVar("Bz0")[0]
        br0, btheta0, bphi0 = spherical_components(
            bx0, by0, bz0, sin_theta, cos_theta, sin_phi, cos_phi
        )

        with h5py.File(output_path, "w") as output:
            write_static_datasets(
                output,
                time_values,
                tiegcm_lat,
                tiegcm_lon,
                inner_lat,
                inner_lon,
                inner_r,
                args,
                tiegcm_path,
            )
            create_output_datasets(
                output,
                n_steps,
                tiegcm_lat.shape,
                inner_lat.shape,
                args.compression,
            )

            for out_step in range(n_steps):
                gamera_step = gsph.s0 + out_step + 1
                event_time = gsph.UT[out_step + 1]
                print(
                    f"Preparing step {out_step + 1} of {n_steps}: "
                    f"{event_time.isoformat()}",
                    flush=True,
                )

                integrated = integrate_tiegcm_step(
                    tiegcm, out_step, args.conductance_source
                )
                for key, values in integrated.items():
                    output[key][out_step] = values

                remix_values = remix_fields_for_step(
                    remix_file,
                    gamera_step,
                    event_time,
                    tiegcm_lon,
                    tiegcm_lat,
                )
                for key, values in remix_values.items():
                    output[key][out_step] = values.astype(np.float32)

                bx = gsph.GetVar("Bx", gamera_step)[0] - bx0
                by = gsph.GetVar("By", gamera_step)[0] - by0
                bz = gsph.GetVar("Bz", gamera_step)[0] - bz0
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gamera-dir",
        default=str(DEFAULT_GAMERA_DIR),
        help="Directory containing the GAMERA run and TIEGCM NetCDF.",
    )
    parser.add_argument(
        "--gamera-subdir",
        default="gamera",
        help="Subdirectory under --gamera-dir containing msphere files.",
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help="GAMERA/REMIX file tag.")
    parser.add_argument("--tiegcm-nc", default=None, help="Explicit TIEGCM NetCDF path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for prepared HDF5 output.",
    )
    parser.add_argument(
        "--output-name", default=DEFAULT_OUTPUT_NAME, help="Prepared HDF5 filename."
    )
    parser.add_argument(
        "--conductance-source",
        choices=("computed", "native"),
        default="computed",
        help=(
            "Use computed vertical integrals for SP/SH, or TIEGCM native "
            "gzigm1/gzigm2 conductances with matching wind numerators."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="HDF5 compression for large time-dependent fields.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Limit steps for a quick test."
    )
    return parser


def main() -> None:
    """Prepare forcing from command-line arguments."""
    output_path = prepare_forcing(build_arg_parser().parse_args())
    print(f"Prepared forcing written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
