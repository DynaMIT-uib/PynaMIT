"""Build reusable MAGE/GAMERA/TIEGCM forcing.

The expensive height integration and source-coordinate transformations
are done here once. The output HDF5 contains the fields used by the
projection step on fixed, Earth-attached geographic grids:

- ``SP`` and ``SH``: Pedersen and Hall conductance in S, with global
  2 S / 1 S background minima for the global PynaMIT sheet.
- ``u_p_theta``/``u_p_phi``: Pedersen-weighted model-basis wind in m/s.
- ``u_h_theta``/``u_h_phi``: Hall-weighted model-basis wind in m/s.
- radial current derived from REMIX FAC and the GAMERA inner-boundary
  radial magnetic perturbation,
  remapped from their timestamped SM source coordinates.

The wind integration intentionally stores conductivity-weighted winds,
not a height-resolved ``u x B`` source. The projection step uses
the PynaMIT sheet-radius main field and sheet resistance, matching the
thin-sheet ``JS -> E_S`` closure.

The prepared file is the minimal projection contract, not a diagnostic
archive. It is written atomically so a failed preparation cannot replace
the last complete forcing file.
"""

from __future__ import annotations

import datetime as dt
import operator
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from kompe.constants import EARTH_RADIUS_M

from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism.kaiju_geopack import kaiju_geopack_sm
from pynamit.workflows.mage.gamera import (
    _centered_dipole_alignment_attrs,
    _datetime_from_mjd,
    _gamera_background_field,
    _gamera_dipole_strength_nT,
    _gamera_inner_boundary_geometry,
    _gamera_internal_dipole_axes,
    _gamera_length_scale_m,
    _GameraBoundaryInterpolator,
    _geographic_grid_in_sm,
    _kaiju_sm_transform_time,
    _pynamit_dipole_B0_T,
    _validate_gamera_dynamic_field_units,
)
from pynamit.workflows.mage.prepared_forcing import (
    CONDUCTANCE_FLOOR_MODEL,
    HALL_CONDUCTANCE_FLOOR_S,
    IONOSPHERE_RADIUS_M,
    MAGE_FORCING_KIND,
    MAGE_FORCING_VERSION,
    MAGE_SOURCE_TIME_TOLERANCE_SECONDS,
    MAGE_TIME_AXIS,
    PEDERSEN_CONDUCTANCE_FLOOR_S,
    TIEGCM_DYNAMO_BOTTOM_ILEV,
    TIEGCM_DYNAMO_REFERENCE_HEIGHT_M,
    TIEGCM_HALL_LOWER_SCALE_M,
    TIEGCM_PEDERSEN_LOWER_SCALE_M,
)
from pynamit.workflows.mage.remix import REMIX_TIME_TOLERANCE_SECONDS, _RemixRadialCurrentReader
from pynamit.workflows.mage.tiegcm import (
    _apply_conductance_floor,
    _integrate_tiegcm_step,
    _resolve_tiegcm_path,
    _tiegcm_times,
    _validate_tiegcm_variables,
)


@dataclass(frozen=True)
class ForcingSettings:
    """Inputs and output policy for one MAGE forcing preparation."""

    gamera_directory: Path
    output_path: Path
    tag: str = "msphere"
    inner_index: int = 0
    tiegcm_path: Path | None = None
    compression: str = "lzf"
    max_steps: int | None = None


# Coupled-source time axis


def _time_offsets_seconds(
    source_times: list[dt.datetime], nominal_times: list[dt.datetime]
) -> np.ndarray:
    """Return signed source-minus-nominal time offsets."""
    return np.array(
        [
            (source_time - nominal_time).total_seconds()
            for source_time, nominal_time in zip(source_times, nominal_times, strict=True)
        ],
        dtype=float,
    )


def _validate_forcing_time_axis(
    nominal_times: list[dt.datetime],
    gamera_times: list[dt.datetime],
    remix_times: list[dt.datetime],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the nominal schedule and exact source times."""
    n_steps = len(nominal_times)
    if n_steps == 0:
        raise RuntimeError("No forcing steps are available.")
    source_lengths = {"GAMERA": len(gamera_times), "ReMIX": len(remix_times)}
    mismatched = {name: size for name, size in source_lengths.items() if size != n_steps}
    if mismatched:
        raise RuntimeError(
            f"The nominal time axis has {n_steps} histories but source counts are {mismatched}."
        )

    time_axes = {"nominal TIEGCM": nominal_times, "GAMERA": gamera_times, "ReMIX": remix_times}
    for name, times in time_axes.items():
        intervals = np.array(
            [(next_time - time).total_seconds() for time, next_time in pairwise(times)],
            dtype=float,
        )
        if np.any(intervals <= 0.0):
            raise RuntimeError(f"The {name} time axis must be strictly increasing.")
        if (
            name == "nominal TIEGCM"
            and intervals.size > 1
            and not np.allclose(intervals, intervals[0], rtol=0.0, atol=1e-9)
        ):
            raise RuntimeError("The nominal TIEGCM time axis must have a uniform cadence.")

    gamera_offsets = _time_offsets_seconds(gamera_times, nominal_times)
    remix_offsets = _time_offsets_seconds(remix_times, nominal_times)
    for source, offsets in {"GAMERA": gamera_offsets, "ReMIX": remix_offsets}.items():
        mismatch = np.flatnonzero(np.abs(offsets) > MAGE_SOURCE_TIME_TOLERANCE_SECONDS)
        if mismatch.size:
            index = int(mismatch[0])
            raise RuntimeError(
                f"{source} is not aligned with the nominal forcing time at source step "
                f"{index}: offset={offsets[index]:g} s; allowed absolute offset is "
                f"{MAGE_SOURCE_TIME_TOLERANCE_SECONDS:g} s."
            )

    remix_gamera_offsets = _time_offsets_seconds(remix_times, gamera_times)
    mismatch = np.flatnonzero(np.abs(remix_gamera_offsets) > REMIX_TIME_TOLERANCE_SECONDS)
    if mismatch.size:
        index = int(mismatch[0])
        raise RuntimeError(
            "ReMIX is not aligned with GAMERA at source step "
            f"{index}: offset={remix_gamera_offsets[index]:g} s; allowed absolute offset is "
            f"{REMIX_TIME_TOLERANCE_SECONDS:g} s."
        )
    return gamera_offsets, remix_offsets


# Prepared-forcing output


def _h5_dataset_kwargs(compression: str) -> dict[str, Any]:
    """Return h5py dataset creation options."""
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


@contextmanager
def _atomic_prepared_output(output_path: Path):
    """Publish a temporary HDF5 file atomically when complete."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}-", suffix=".tmp.h5", dir=output_path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with h5py.File(temporary_path, "w") as output:
            yield output
            output.attrs["complete"] = True
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _create_output_datasets(
    output: h5py.File,
    n_steps: int,
    ion_shape: tuple[int, int],
    inner_shape: tuple[int, int],
    compression: str,
) -> None:
    """Create all time-dependent output datasets."""
    kwargs = _h5_dataset_kwargs(compression)
    for name in ("jr", "SH", "SP", "u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi"):
        output.create_dataset(name, shape=(n_steps, *ion_shape), dtype="f4", **kwargs)
    output.create_dataset("delta_Br", shape=(n_steps, *inner_shape), dtype="f4", **kwargs)
    output["jr"].attrs["units"] = "uA m-2"
    output["jr"].attrs["description"] = (
        "outward radial current from upward-positive REMIX FAC times abs(source unit_br); "
        "zero outside REMIX coverage"
    )
    output["SP"].attrs["units"] = "S"
    output["SP"].attrs["description"] = (
        "radially integrated TIEGCM Pedersen conductance with a global hard 2 S "
        "background minimum for the PynaMIT sheet"
    )
    output["SH"].attrs["units"] = "S"
    output["SH"].attrs["description"] = (
        "radially integrated TIEGCM Hall conductance with a global hard 1 S "
        "background minimum for the PynaMIT sheet"
    )
    for name in ("u_p_theta", "u_p_phi", "u_h_theta", "u_h_phi"):
        output[name].attrs["units"] = "m s-1"
    output["delta_Br"].attrs["units"] = "nT"
    output["delta_Br"].attrs["description"] = (
        "radial perturbation from cell-volume-average total B minus the matching "
        "cell-volume-average GAMERA split background B0"
    )


def _write_time_axis(
    output: h5py.File,
    nominal_times: list[dt.datetime],
    gamera_times: list[dt.datetime],
    remix_times: list[dt.datetime],
) -> None:
    """Write the nominal clock and exact coupled-source times."""
    string_dtype = h5py.string_dtype(encoding="utf-8")
    timestamp_datasets = {
        "time": nominal_times,
        "gamera_source_time": gamera_times,
        "remix_source_time": remix_times,
    }
    for name, times in timestamp_datasets.items():
        values = np.asarray([value.isoformat() for value in times], dtype=string_dtype)
        output.create_dataset(name, data=values, dtype=string_dtype)

    output["time"].attrs["description"] = (
        "nominal forcing application time from the uniform TIEGCM mtime schedule"
    )
    output["gamera_source_time"].attrs["description"] = (
        "exact GAMERA history time retained as provenance; Kaiju SM transformations "
        "round it to the nearest whole second"
    )
    output["remix_source_time"].attrs["description"] = "exact coupled ReMIX history time"

    for source, times in {"gamera": gamera_times, "remix": remix_times}.items():
        name = f"{source}_time_offset_seconds"
        output.create_dataset(name, data=_time_offsets_seconds(times, nominal_times))
        output[name].attrs["units"] = "s"
        output[name].attrs["description"] = f"{source.upper()} source time minus nominal time"

    output.attrs["time_axis"] = MAGE_TIME_AXIS
    output.attrs["source_time_tolerance_seconds"] = MAGE_SOURCE_TIME_TOLERANCE_SECONDS


def _write_static_datasets(
    output: h5py.File,
    gamera_reference_time: dt.datetime,
    ionosphere_lat: np.ndarray,
    ionosphere_lon: np.ndarray,
    inner_lat: np.ndarray,
    inner_lon: np.ndarray,
    inner_r: np.ndarray,
    inner_solid_angle: np.ndarray,
    settings: ForcingSettings,
    gamera_directory: Path,
    length_scale_m: float,
    mag_m0_nT: float,
    tiegcm_path: Path,
    remix_equatorward_sm_latitude: float,
) -> None:
    """Write static datasets and metadata."""
    output.attrs["kind"] = MAGE_FORCING_KIND
    output.attrs["version"] = MAGE_FORCING_VERSION
    output.attrs["complete"] = False
    output.create_dataset("ionosphere_lat", data=ionosphere_lat)
    output.create_dataset("ionosphere_lon", data=ionosphere_lon)
    output.create_dataset("boundary_lat", data=inner_lat)
    output.create_dataset("boundary_lon", data=inner_lon)
    output.create_dataset("boundary_radius", data=inner_r)
    output.create_dataset("boundary_solid_angle", data=inner_solid_angle)
    for name in ("ionosphere_lat", "ionosphere_lon", "boundary_lat", "boundary_lon"):
        output[name].attrs["units"] = "degree"
    output["boundary_radius"].attrs["units"] = "m"
    output["boundary_radius"].attrs["description"] = (
        "radius of the Kaiju volume-barycentric GAMERA boundary cell center"
    )
    output["boundary_solid_angle"].attrs["units"] = "sr"
    output["boundary_solid_angle"].attrs["description"] = (
        "cell solid angle from the true GAMERA inner-boundary vertices"
    )
    output.attrs["gamera_directory"] = str(gamera_directory)
    output.attrs["tiegcm_nc"] = str(tiegcm_path)
    output.attrs["tiegcm_conductance_integration"] = (
        "radial_geometric_height_with_lower_dynamo_extension"
    )
    output.attrs["tiegcm_dynamo_bottom_ilev"] = TIEGCM_DYNAMO_BOTTOM_ILEV
    output.attrs["tiegcm_dynamo_reference_height_m"] = TIEGCM_DYNAMO_REFERENCE_HEIGHT_M
    output.attrs["tiegcm_pedersen_lower_scale_m"] = TIEGCM_PEDERSEN_LOWER_SCALE_M
    output.attrs["tiegcm_hall_lower_scale_m"] = TIEGCM_HALL_LOWER_SCALE_M
    output.attrs["conductance_floor_model"] = CONDUCTANCE_FLOOR_MODEL
    output.attrs["pedersen_conductance_floor_S"] = PEDERSEN_CONDUCTANCE_FLOOR_S
    output.attrs["hall_conductance_floor_S"] = HALL_CONDUCTANCE_FLOOR_S
    output.attrs["remix_grid_equatorward_sm_latitude_deg"] = float(remix_equatorward_sm_latitude)
    output.attrs["tiegcm_vertical_grid"] = (
        "SIGMA_PED/SIGMA_HAL and UN/VN at lev[:-1], with dz=diff(ZG at ilev); "
        "terminal fill-only lev omitted; below the first saved interface, conductivity is "
        "continued against Z to ilev=-8.5 at 90 km using TIEGCM pdynamo scale lengths, "
        "radial thickness uses the corresponding ZG intervals, and the lowest winds are "
        "held constant"
    )
    output.attrs["coordinate_system"] = "GEO"
    output.attrs["longitude_convention"] = "east_positive_degrees"
    output.attrs["tiegcm_source_coordinate_system"] = "geographic"
    output.attrs["ionosphere_radius_m"] = IONOSPHERE_RADIUS_M
    output.attrs["wind_weighting"] = (
        "u_p = integral(sigma_P*u*dr)/SP and u_h = integral(sigma_H*u*dr)/SH; "
        "components are geographic south/east on the native TIEGCM grid. Where a "
        "global background minimum raises a conductance, its unresolved conductivity "
        "is assumed to share the corresponding TIEGCM conductivity-weighted mean wind"
    )
    output.attrs["remix_tag"] = settings.tag
    output.attrs["fac_convention"] = "upward"
    output.attrs["fac_source"] = (
        "Kaiju ReMIX Field-aligned current NORTH/SOUTH, converted from "
        "parallel-positive to upward-positive"
    )
    output.attrs["radial_current_convention"] = "outward"
    output.attrs["fac_to_radial_current"] = "jr = FAC_upward * abs(source unit_br)"
    output.attrs["remix_fac_interpolation"] = "kaiju_native_periodic"
    output.attrs["gamera_boundary_interpolation"] = (
        "gamera_native_periodic_bilinear_with_polar_mean"
    )
    output.attrs["gamera_sm_transform_time_convention"] = "kaiju_mjdrecalc_nearest_second"
    output.attrs["gamera_inner_index"] = int(settings.inner_index)
    output.attrs["gamera_length_scale_m"] = float(length_scale_m)
    output.attrs["gamera_background_reference"] = "cell_volume_average_split_B0"
    output.attrs["gamera_B_output"] = (
        "Kaiju cell-volume-average total Bx/By/Bz; delta_Br removes the matching "
        "cell-volume-average split B0, not point-sampled BxD/ByD/BzD"
    )
    for name, value in _centered_dipole_alignment_attrs(gamera_reference_time, mag_m0_nT).items():
        output.attrs[name] = value
    output.attrs["gamera_mag_m0_nT"] = float(mag_m0_nT)
    output.attrs["main_field_B0_T"] = _pynamit_dipole_B0_T(mag_m0_nT, length_scale_m)
    output.attrs["main_field_B0_reference_radius_m"] = EARTH_RADIUS_M


def _validate_settings(settings: ForcingSettings) -> None:
    """Validate MAGE preparation settings."""
    if settings.compression not in ("lzf", "gzip", "none"):
        raise ValueError(
            f"compression must be 'lzf', 'gzip', or 'none'; got {settings.compression!r}."
        )
    if isinstance(settings.inner_index, (bool, np.bool_)):
        raise ValueError("inner_index must be a non-negative integer.")
    try:
        inner_index = operator.index(settings.inner_index)
    except TypeError as exc:
        raise ValueError("inner_index must be a non-negative integer.") from exc
    if inner_index < 0:
        raise ValueError(f"inner_index must be non-negative; got {settings.inner_index}.")
    if settings.max_steps is not None:
        if isinstance(settings.max_steps, (bool, np.bool_)):
            raise ValueError("max_steps must be a positive integer.")
        try:
            max_steps = operator.index(settings.max_steps)
        except TypeError as exc:
            raise ValueError("max_steps must be a positive integer.") from exc
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive; got {settings.max_steps}.")


def prepare_forcing(settings: ForcingSettings) -> Path:
    """Prepare the HDF5 forcing file."""
    _validate_settings(settings)
    from netCDF4 import Dataset

    try:
        import kaipy.gamera.magsphere as msph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MAGE forcing preparation needs kaipy to read GAMERA files. "
            f"Missing module: {exc.name!r}. Run it in the MAGE/GAMERA environment "
            "where kaipy and its dependencies are installed."
        ) from exc

    gamera_directory = Path(settings.gamera_directory).expanduser()
    if not gamera_directory.is_dir():
        raise FileNotFoundError(f"GAMERA directory does not exist: {gamera_directory}")
    tiegcm_path = _resolve_tiegcm_path(gamera_directory, settings.tiegcm_path)
    remix_file = gamera_directory / f"{settings.tag}.mix.h5"
    if not remix_file.is_file():
        raise FileNotFoundError(f"REMIX file does not exist: {remix_file}")
    output_path = Path(settings.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using GAMERA directory: {gamera_directory}", flush=True)
    print(f"Using TIEGCM file: {tiegcm_path}", flush=True)
    print(f"Using REMIX file: {remix_file}", flush=True)
    print(f"Writing prepared forcing: {output_path}", flush=True)

    gsph = msph.GamsphPipe(str(gamera_directory), settings.tag, doFast=False)
    if settings.inner_index >= gsph.X.shape[0] - 1:
        raise ValueError(
            f"inner_index must be between 0 and {gsph.X.shape[0] - 2}; got {settings.inner_index}."
        )
    length_scale_m = _gamera_length_scale_m(gsph)
    mag_m0_nT = _gamera_dipole_strength_nT(gsph)
    bx0, by0, bz0 = _gamera_background_field(gsph, settings.inner_index)
    print(f"Using GAMERA length scale: {length_scale_m:.6g} m", flush=True)
    axes = _gamera_internal_dipole_axes(mag_m0_nT)
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
    if not getattr(gsph, "hasMJD", False):
        raise RuntimeError("GAMERA forcing must provide MJD time metadata.")
    n_steps = len(gsph.MJDs) - 1
    if settings.max_steps is not None:
        n_steps = min(n_steps, int(settings.max_steps))
    if n_steps <= 0:
        raise RuntimeError("GAMERA contains no forcing steps after its initial state.")
    _validate_gamera_dynamic_field_units(gsph, gsph.s0 + 1)

    with Dataset(tiegcm_path, mode="r") as tiegcm:
        _validate_tiegcm_variables(tiegcm, n_steps)
        gamera_times = [_datetime_from_mjd(value) for value in gsph.MJDs[1 : n_steps + 1]]
        nominal_times = _tiegcm_times(tiegcm, gamera_times)
        source_lon = np.asarray(tiegcm.variables["lon"][:], dtype=float)
        source_lat = np.asarray(tiegcm.variables["lat"][:], dtype=float)
        ionosphere_lon, ionosphere_lat = np.meshgrid(wrap_longitude_180(source_lon), source_lat)

        boundary = _gamera_inner_boundary_geometry(gsph, settings.inner_index, length_scale_m)
        boundary_lat, boundary_lon = kaiju_geopack_sm(
            _kaiju_sm_transform_time(gamera_times[0])
        ).sm2geo(boundary.sm_latitude, boundary.sm_longitude)
        boundary_interpolator = _GameraBoundaryInterpolator(
            boundary.sm_latitude, boundary.sm_longitude
        )
        # Kaiju gioH5 writes Bx/By/Bz as total field when
        # Model%doBackground is true, and root Bx0/By0/Bz0 as Gr%B0.
        # PynaMIT needs the perturbation.
        with _RemixRadialCurrentReader(remix_file) as radial_current_reader:
            remix_equatorward_sm_latitude = radial_current_reader.equatorward_sm_latitude
            print(
                "Applying global PynaMIT sheet-conductance floors: "
                f"Pedersen {PEDERSEN_CONDUCTANCE_FLOOR_S:g} S, "
                f"Hall {HALL_CONDUCTANCE_FLOOR_S:g} S",
                flush=True,
            )
            gamera_steps = [gsph.s0 + out_step + 1 for out_step in range(n_steps)]
            remix_times = [
                radial_current_reader.source_time(gamera_step) for gamera_step in gamera_steps
            ]
            gamera_offsets, remix_offsets = _validate_forcing_time_axis(
                nominal_times, gamera_times, remix_times
            )
            print(
                "Canonical forcing clock: TIEGCM mtime; "
                f"GAMERA offsets {gamera_offsets.min():.6g} to {gamera_offsets.max():.6g} s; "
                f"ReMIX offsets {remix_offsets.min():.6g} to {remix_offsets.max():.6g} s",
                flush=True,
            )

            with _atomic_prepared_output(output_path) as output:
                _write_time_axis(output, nominal_times, gamera_times, remix_times)
                _write_static_datasets(
                    output,
                    gamera_times[0],
                    ionosphere_lat,
                    ionosphere_lon,
                    boundary_lat,
                    boundary_lon,
                    boundary.radius_m,
                    boundary.solid_angle,
                    settings,
                    gamera_directory,
                    length_scale_m,
                    mag_m0_nT,
                    tiegcm_path,
                    remix_equatorward_sm_latitude,
                )
                _create_output_datasets(
                    output, n_steps, ionosphere_lat.shape, boundary_lat.shape, settings.compression
                )

                for out_step, (gamera_step, gamera_time) in enumerate(
                    zip(gamera_steps, gamera_times, strict=True)
                ):
                    print(
                        f"Preparing step {out_step + 1} of {n_steps}: "
                        f"nominal {nominal_times[out_step].isoformat()}, "
                        f"GAMERA {gamera_time.isoformat()}",
                        flush=True,
                    )

                    integrated = _integrate_tiegcm_step(tiegcm, out_step)
                    integrated["SP"], integrated["SH"] = _apply_conductance_floor(
                        integrated["SP"], integrated["SH"]
                    )
                    for key, values in integrated.items():
                        output[key][out_step] = values

                    output["jr"][out_step] = radial_current_reader.read(
                        gamera_step, ionosphere_lon, ionosphere_lat, gamera_time
                    ).astype(np.float32)

                    bx = gsph.GetVar("Bx", gamera_step)[settings.inner_index] - bx0
                    by = gsph.GetVar("By", gamera_step)[settings.inner_index] - by0
                    bz = gsph.GetVar("Bz", gamera_step)[settings.inner_index] - bz0
                    delta_br_sm = boundary.radial_component(bx, by, bz)
                    boundary_sm_lat, boundary_sm_lon = _geographic_grid_in_sm(
                        boundary_lat, boundary_lon, gamera_time
                    )
                    output["delta_Br"][out_step] = boundary_interpolator.interpolate(
                        delta_br_sm, target_sm_lat=boundary_sm_lat, target_sm_lon=boundary_sm_lon
                    ).astype(np.float32)

    return output_path


__all__ = ["ForcingSettings", "prepare_forcing"]
