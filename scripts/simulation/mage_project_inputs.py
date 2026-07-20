"""Project prepared MAGE/GAMERA/TIEGCM forcing into PynaMIT inputs.

Run ``mage_prepare_forcing.py`` first if the height-integrated HDF5 file
does not exist.  This script does the PynaMIT-specific projection step:
it reads the prepared HDF5 file, projects Br, jr, conductance, and the
direct Pedersen/Hall weighted wind electric-field source, then writes a
reusable input package under
``scripts/simulation/mage_runs/mage_2011_kaiju_direct_e/projections``.

The time evolution is intentionally not done here.  Use
``mage_forcing_final.py`` to run PynaMIT from the projected input
package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import pynamit
from pynamit.coordinates import wrap_longitude_180
from pynamit.geomagnetism import MagneticFieldEvaluation, MainField, decimal_year
from pynamit.simulation.workflows.mage import (
    DEFAULT_MMAX,
    DEFAULT_NCS,
    DEFAULT_NMAX,
    area_sqrt_weights,
    boundary_radius_from_h5,
    dipole_B0_from_h5,
    dipole_radial_sampling,
    direct_E_source_for_pynamit,
    file_fingerprint,
    gamera_internal_dipole_details,
    h5_time_vector_seconds,
    load_weighted_winds,
    projection_directory_for_resolution,
    result_directory_for_resolution,
    summarize_input_cadence,
    tangential_sqrt_weights,
)
from pynamit.simulation.workflows.prepared_inputs import (
    clear_prepared_input_package,
    write_input_manifest,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RE = 6381e3
RI = 6.5e6
LATITUDE_BOUNDARY = 35.0

BR_LAMBDA = 0.1
CONDUCTANCE_LAMBDA = 3.0
JR_LAMBDA = 0.1
E_SOURCE_LAMBDA = 0.1

MAGE_BR_LOCAL_NOON_LONGITUDE = 0.0
CENTERED_DIPOLE_MODELS = ("kaiju_dipole", "dipole")

DEFAULT_FORCING_CANDIDATES = (
    SCRIPT_DIR / "mage_prepared" / "mage_prepared_forcing.h5",
    SCRIPT_DIR / "mage_prepared" / "data_H_int_qeff.h5",
    Path("/disk/Gamera_Dong/prep_Pynamit/mage_prepared_forcing.h5"),
    Path("/disk/Gamera_Dong/prep_Pynamit/data_H_int_qeff.h5"),
    Path("/disk/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
    Path("~/Gamera_Dong/prep_Pynamit/mage_prepared_forcing.h5"),
    Path("~/Gamera_Dong/prep_Pynamit/data_H_int_qeff.h5"),
    Path("mage_2011/data_H_int.h5"),
    Path("~/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
    Path("/Users/andreasskeidsvoll/Gamera_Dong/prep_Pynamit/data_H_int.h5"),
)
DEFAULT_MAGE_RUN_ROOT = SCRIPT_DIR / "mage_runs" / "mage_2011_kaiju_direct_e"
MAGE_INPUT_METADATA_FILENAME = "mage_input_metadata.json"


DEFAULT_INPUT_DIRECTORY = projection_directory_for_resolution(
    DEFAULT_NMAX, DEFAULT_MMAX, DEFAULT_NCS, DEFAULT_MAGE_RUN_ROOT
)
DEFAULT_RESULT_DIRECTORY = result_directory_for_resolution(
    DEFAULT_NMAX, DEFAULT_MMAX, DEFAULT_NCS, DEFAULT_MAGE_RUN_ROOT
)


@dataclass(frozen=True)
class MageInputProjectionSettings:
    """Defaults intended to be edited for the MAGE projection step."""

    forcing_h5: Path | None = None
    input_directory: Path | None = None
    main_field_kind: str = "kaiju_dipole"
    dipole_B0: float | None = None
    fac_convention: str = "upward"
    RM: float | None = None
    nmax: int = DEFAULT_NMAX
    mmax: int = DEFAULT_MMAX
    ncs: int = DEFAULT_NCS
    max_steps: int | None = None
    br_lambda: float = BR_LAMBDA
    conductance_lambda: float = CONDUCTANCE_LAMBDA
    jr_lambda: float = JR_LAMBDA
    e_source_lambda: float = E_SOURCE_LAMBDA
    area_weighted_least_squares: bool = False
    artifact_storage: str = "auto"


SETTINGS = MageInputProjectionSettings()


def resolve_existing_path(
    path: str | Path | None, candidates: tuple[Path, ...], label: str
) -> Path:
    """Resolve an explicit path or the first existing candidate."""
    if path:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"{label} does not exist: {resolved}")
        return resolved

    for candidate in candidates:
        if candidate.expanduser().exists():
            return candidate.expanduser()

    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked: {formatted}")


def projected_resistance_values(
    simulation: pynamit.Simulation, grid: pynamit.Grid, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate projected sheet resistance coefficients on ``grid``."""
    conductance_entry = simulation.run_data.input_series.get_entry("resistance", time)
    if conductance_entry is None:
        raise RuntimeError("Conductance must be set before computing direct wind E_source.")
    field_space = simulation.run_data.schema.input_field_spaces["resistance"]
    evaluator = field_space.representation.get_scalar_evaluation_operator(grid)
    return (
        np.asarray(evaluator.matvec(conductance_entry["etaP"])).reshape(-1),
        np.asarray(evaluator.matvec(conductance_entry["etaH"])).reshape(-1),
    )


def print_field_stats(label: str, values: np.ndarray) -> None:
    """Print compact finite-value diagnostics."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        print(f"{label}: no finite values", flush=True)
        return
    sample = values[finite]
    print(
        f"{label}: min={sample.min():.6g}, rms={np.sqrt(np.mean(sample**2)):.6g}, "
        f"max={sample.max():.6g}",
        flush=True,
    )


def _json_value(value: Any) -> Any:
    """Return a JSON-serializable metadata value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_mage_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write MAGE-specific input-package metadata."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8"
    )


def _clear_existing_input_package(directory: Path, artifact_storage: str) -> None:
    """Clear the package and MAGE sidecar before reprojection."""
    artifact_names = clear_prepared_input_package(directory, artifact_storage=artifact_storage)
    if artifact_names:
        names = ", ".join(artifact_names)
        print(f"Replacing existing PynaMIT artifacts: {names}", flush=True)
    (directory / MAGE_INPUT_METADATA_FILENAME).unlink(missing_ok=True)


def project_mage_inputs(settings: MageInputProjectionSettings = SETTINGS) -> Path:
    """Project configured MAGE inputs into a PynaMIT input package."""
    if settings.main_field_kind not in CENTERED_DIPOLE_MODELS:
        raise ValueError(
            f"Unsupported main_field_kind {settings.main_field_kind!r}; "
            f"expected one of {CENTERED_DIPOLE_MODELS}."
        )
    if settings.fac_convention not in ("upward", "field_aligned"):
        raise ValueError(
            "fac_convention must be either 'upward' or 'field_aligned'; "
            f"got {settings.fac_convention!r}."
        )

    h5_path = resolve_existing_path(
        settings.forcing_h5, DEFAULT_FORCING_CANDIDATES, "forcing HDF5"
    )

    import h5py

    input_directory = Path(
        settings.input_directory
        or projection_directory_for_resolution(
            settings.nmax, settings.mmax, settings.ncs, DEFAULT_MAGE_RUN_ROOT
        )
    ).expanduser()
    input_directory.mkdir(parents=True, exist_ok=True)
    _clear_existing_input_package(input_directory, settings.artifact_storage)

    with h5py.File(h5_path, "r") as file:
        forcing_times, input_times = h5_time_vector_seconds(file["time"][:])
        event_time = forcing_times[0]
        coordinate_time = event_time
        dipole_epoch = decimal_year(event_time)
        RM = boundary_radius_from_h5(file, settings.RM)
        dipole_B0 = dipole_B0_from_h5(file, settings.dipole_B0)
        main_field = MainField(kind=settings.main_field_kind, epoch=dipole_epoch, B0=dipole_B0)
        gamera_dipole = gamera_internal_dipole_details(file)
        alignment = main_field.alignment_metadata(event_time)
        rk = dipole_radial_sampling(RI, RM, n_steps=40)

        ionosphere_lat_geo = np.asarray(file["glat"][:], dtype=float)
        ionosphere_lon_geo = wrap_longitude_180(file["glon"][:])
        ionosphere_lat, ionosphere_lon = main_field.geo_to_model_coordinates(
            ionosphere_lat_geo, ionosphere_lon_geo, event_time=coordinate_time
        )
        ionosphere_grid = pynamit.Grid(lat=ionosphere_lat, lon=ionosphere_lon)

        magnetosphere_lat_raw = np.asarray(file["Blat"][:], dtype=float)
        magnetosphere_lon_raw = np.asarray(file["Blon"][:], dtype=float)
        magnetosphere_lon = main_field.local_time_longitude_to_model_longitude(
            magnetosphere_lon_raw,
            coordinate_time,
            local_noon_longitude=MAGE_BR_LOCAL_NOON_LONGITUDE,
        )
        magnetosphere_grid = pynamit.Grid(lat=magnetosphere_lat_raw, lon=magnetosphere_lon)
        magnetosphere_sqrt_weights = area_sqrt_weights(magnetosphere_grid.lat)
        ionosphere_sqrt_weights = area_sqrt_weights(ionosphere_grid.lat)
        ionosphere_tangential_sqrt_weights = tangential_sqrt_weights(ionosphere_grid.lat)

        print(f"Using forcing file: {h5_path}", flush=True)
        print(f"Writing projected input package: {input_directory}", flush=True)
        print(f"Event time: {event_time.isoformat()}", flush=True)
        print(f"Coordinate time: {coordinate_time.isoformat()}", flush=True)
        print(
            "Forcing time span: "
            f"{forcing_times[0].isoformat()} to {forcing_times[-1].isoformat()} "
            f"({len(forcing_times)} step(s))",
            flush=True,
        )
        print(f"Main field used for projection: {settings.main_field_kind}", flush=True)
        print(f"Dipole alignment model: {alignment['dipole_alignment_model']}", flush=True)
        print(f"Dipole epoch: {dipole_epoch:.9f}", flush=True)
        print(f"Dipole B0: {dipole_B0:.6g} T ({dipole_B0 * 1e9:.6g} nT)", flush=True)
        print(f"GAMERA signed MagM0: {gamera_dipole['mag_m0_nT']:.6g} nT", flush=True)
        print("GAMERA coordinates: SM; REMIX longitude 0 = noon", flush=True)
        print(f"RM: {RM:.6g} m", flush=True)
        print("Wind forcing: direct E_source from Pedersen/Hall weighted winds", flush=True)
        print(f"FAC convention: {settings.fac_convention}", flush=True)

        simulation = pynamit.Simulation(
            run_directory=input_directory,
            Nmax=settings.nmax,
            Mmax=settings.mmax,
            Ncs=settings.ncs,
            RI=RI,
            RM=RM,
            magnetic_boundary_shielding=False,
            main_field_kind=settings.main_field_kind,
            main_field_epoch=dipole_epoch,
            main_field_B0=dipole_B0,
            fac_integration_radii=rk,
            enable_pfac_coupling=False,
            enable_interhemispheric_coupling=False,
            interhemispheric_coupling_latitude=LATITUDE_BOUNDARY,
            interhemispheric_electric_field_weight=1e-5,
            t0=str(event_time),
            save_steady_states=False,
            integrator="exponential",
            area_weighted_least_squares=settings.area_weighted_least_squares,
            artifact_storage=settings.artifact_storage,
        )

        fac_field_evaluation = MagneticFieldEvaluation(
            simulation.geometry.main_field, ionosphere_grid, RI
        )
        wind_field_evaluation = MagneticFieldEvaluation(
            simulation.geometry.main_field, ionosphere_grid, RI
        )

        if settings.max_steps is not None:
            input_times = input_times[: int(settings.max_steps)]
            forcing_times = forcing_times[: int(settings.max_steps)]
        n_steps = input_times.size
        if n_steps == 0:
            raise ValueError("No forcing time steps selected for projection.")

        for step in range(n_steps):
            input_time = float(input_times[step])
            print(
                f"Projecting input step {step + 1} of {n_steps} "
                f"at t={input_time:g} s ({forcing_times[step].isoformat()})",
                flush=True,
            )

            delta_Br = np.asarray(file["Bu"][step], dtype=float).reshape(-1) * 1e-9
            if np.any(~np.isfinite(delta_Br)):
                raise ValueError("Br input contains non-finite values.")
            print_field_stats("  Delta Br [T]", delta_Br)
            simulation.set_Br(
                delta_Br,
                lat=magnetosphere_grid.lat,
                lon=magnetosphere_grid.lon,
                time=input_time,
                sqrt_weights=magnetosphere_sqrt_weights,
                reg_lambda=settings.br_lambda,
            )

            FAC = np.asarray(file["FAC"][step], dtype=float) * 1e-6
            if np.any(~np.isfinite(FAC)):
                print("  FAC contains non-finite values; setting them to 0.", flush=True)
                FAC[~np.isfinite(FAC)] = 0.0
            if settings.fac_convention == "field_aligned":
                jr = FAC.reshape(-1) * fac_field_evaluation.unit_br
            else:
                jr = FAC.reshape(-1)
            print_field_stats("  jr [A/m^2]", jr)
            simulation.set_jr(
                jr,
                lat=ionosphere_grid.lat,
                lon=ionosphere_grid.lon,
                time=input_time,
                sqrt_weights=ionosphere_sqrt_weights,
                reg_lambda=settings.jr_lambda,
            )

            sigma_h = np.asarray(file["SH"][step], dtype=float).reshape(-1)
            sigma_p = np.asarray(file["SP"][step], dtype=float).reshape(-1)
            if np.any(~np.isfinite(sigma_h)) or np.any(sigma_h <= 0.0):
                raise ValueError("Hall conductance contains non-finite or non-positive values.")
            if np.any(~np.isfinite(sigma_p)) or np.any(sigma_p <= 0.0):
                raise ValueError(
                    "Pedersen conductance contains non-finite or non-positive values."
                )
            print_field_stats("  Hall conductance [S]", sigma_h)
            print_field_stats("  Pedersen conductance [S]", sigma_p)
            simulation.set_conductance(
                sigma_h,
                sigma_p,
                lat=ionosphere_grid.lat,
                lon=ionosphere_grid.lon,
                time=input_time,
                sqrt_weights=ionosphere_sqrt_weights,
                reg_lambda=settings.conductance_lambda,
            )

            u_p_east, u_p_north, u_h_east, u_h_north = load_weighted_winds(file, step)
            _, _, u_p_east, u_p_north = main_field.geo_to_model_coordinates(
                ionosphere_lat_geo,
                ionosphere_lon_geo,
                u_p_east,
                u_p_north,
                event_time=coordinate_time,
            )
            u_p_theta = -np.asarray(u_p_north, dtype=float).reshape(-1)
            u_p_phi = np.asarray(u_p_east, dtype=float).reshape(-1)
            print_field_stats("  Pedersen-weighted wind speed [m/s]", np.hypot(u_p_theta, u_p_phi))

            _, _, u_h_east, u_h_north = main_field.geo_to_model_coordinates(
                ionosphere_lat_geo,
                ionosphere_lon_geo,
                u_h_east,
                u_h_north,
                event_time=coordinate_time,
            )
            u_h_theta = -np.asarray(u_h_north, dtype=float).reshape(-1)
            u_h_phi = np.asarray(u_h_east, dtype=float).reshape(-1)
            print_field_stats("  Hall-weighted wind speed [m/s]", np.hypot(u_h_theta, u_h_phi))

            eta_p, eta_h = projected_resistance_values(simulation, ionosphere_grid, input_time)
            e_source_theta, e_source_phi = direct_E_source_for_pynamit(
                sigma_p=sigma_p,
                sigma_h=sigma_h,
                u_p_theta=u_p_theta,
                u_p_phi=u_p_phi,
                u_h_theta=u_h_theta,
                u_h_phi=u_h_phi,
                field=wind_field_evaluation,
                eta_p=eta_p,
                eta_h=eta_h,
            )
            print_field_stats(
                "  Direct wind E_source [V/m]", np.hypot(e_source_theta, e_source_phi)
            )
            simulation.set_E_source(
                E_source_theta=e_source_theta,
                E_source_phi=e_source_phi,
                lat=ionosphere_grid.lat,
                lon=ionosphere_grid.lon,
                time=input_time,
                sqrt_weights=ionosphere_tangential_sqrt_weights,
                reg_lambda=settings.e_source_lambda,
            )

        projected_datasets = [
            key
            for key in simulation.run_data.schema.input_variables
            if key in simulation.run_data.input_series.datasets
        ]
        source_tiegcm = file.attrs.get("tiegcm_nc", None)
        if isinstance(source_tiegcm, bytes):
            source_tiegcm = source_tiegcm.decode("utf-8", errors="replace")
        source_files = {
            "forcing_h5": file_fingerprint(h5_path),
            "tiegcm_nc": None if source_tiegcm is None else str(source_tiegcm),
        }
        input_time_metadata = {
            "source_time_first": forcing_times[0].isoformat(),
            "source_time_last": forcing_times[-1].isoformat(),
            "input_time_first_s": float(input_times[0]),
            "input_time_last_s": float(input_times[-1]),
            **summarize_input_cadence(input_times),
        }
        write_input_manifest(
            input_directory,
            simulation.run_data.config,
            input_datasets=projected_datasets,
            source="mage_project_inputs.py",
            notes=(
                "MAGE direct E_source was computed from Pedersen/Hall weighted winds "
                "using the sheet-radius main field and projected sheet resistance.",
            ),
            metadata={
                "input_kind": "mage_gamera_tiegcm",
                "forcing_h5": str(h5_path),
                "tiegcm_nc": None if source_tiegcm is None else str(source_tiegcm),
                "event_time": event_time.isoformat(),
                "coordinate_time": coordinate_time.isoformat(),
                "fac_convention": settings.fac_convention,
                "source_files": source_files,
                **input_time_metadata,
            },
        )
        _write_mage_metadata(
            input_directory / MAGE_INPUT_METADATA_FILENAME,
            {
                "forcing_h5": h5_path,
                "tiegcm_nc": None if source_tiegcm is None else str(source_tiegcm),
                "event_time": event_time.isoformat(),
                "coordinate_time": coordinate_time.isoformat(),
                "main_field_kind": settings.main_field_kind,
                "main_field_epoch": dipole_epoch,
                "dipole_B0_T": dipole_B0,
                "RM_m": RM,
                "fac_convention": settings.fac_convention,
                "n_projected_steps": n_steps,
                **input_time_metadata,
                "projected_datasets": projected_datasets,
                "source_files": source_files,
                "gamera_dipole": gamera_dipole,
                "alignment": alignment,
            },
        )

    print(f"Projected input package written to {input_directory}", flush=True)
    return input_directory


def main(settings: MageInputProjectionSettings = SETTINGS) -> None:
    """Project MAGE inputs from in-script settings."""
    project_mage_inputs(settings)


if __name__ == "__main__":
    main()
