"""Diagnose onset-sign response for the ``mage_forcing_3.py`` configuration.

This script is meant to answer a narrow question: for one selected raw MAGE
sample, do the source ingestion signs and the early low-latitude magnetic
response have the expected sign family?

It does three things from the raw HDF5 file and one benchmark-mode Dynamics
configuration loaded from an existing ``mage_forcing_3.py`` run directory:

1. Convert the selected raw source sample using the same conventions as
   ``mage_forcing_3.py``:
   - ``FAC`` is treated as upward-positive in the file and flipped in the
     Northern Hemisphere before ``set_FAC(...)``.
   - ``Bu`` is passed directly to ``set_Br(...)``.
   - ``We``/``Wn`` become ``u_phi = We`` and ``u_theta = -Wn``.
2. Report local neighborhood means of the raw sources around stations and a
   few synthetic low-latitude local-time samples.
3. Run constant-forcing onset responses for ``jr_only``, ``br_only``,
   ``wind_only``, ``jr_plus_br``, and ``all_sources`` with
   ``connect_hemispheres=False/True`` and report:
   - initial induced ground-field tendency from the forcing term
   - finite-time induced ground response at requested times

The output is JSON so it can be compared between runs or saved to disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py as h5
import numpy as np

import pynamit
from pynamit.postprocess.ground_response import build_ground_magnetic_response_operators
from pynamit.primitives.grid import Grid
from pynamit.simulation.input import compute_spherical_input_sqrt_weights
from pynamit.simulation.mage_time_window import select_mage_time_window
from pynamit.simulation.settings import DynamicsMode, ExponentialSolverKind

DEFAULT_H5_PATH = Path(__file__).resolve().parent / "mage_2011" / "data_H_int.h5"
DEFAULT_RUN_DIRECTORY = (
    Path(__file__).resolve().parent / "results_mage_2011_full_induction_182000_184500"
)
DEFAULT_TIME = "2011-10-24 18:31:00"
DEFAULT_TIMES = (10.0, 60.0, 120.0)
DEFAULT_LATITUDES = (-30.0, -15.0, 15.0, 30.0)
DEFAULT_LONGITUDE_STEP = 30.0
DEFAULT_NEIGHBORHOOD_RADIUS_DEG = 12.0

IONO_WEIGHTING = "mw"
BR_WEIGHTING = "geom_area"
NMAX = 50

BASE_STATIONS = {
    "IPM": (65.136, -147.478),
    "KDU": (-12.693, 132.471),
}

LOCAL_TIME_SAMPLES = {
    "midnight": 0.0,
    "dawn": 6.0,
    "noon": 12.0,
    "dusk": 18.0,
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _normalize_lon(lon_deg: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(lon_deg) + 180.0) % 360.0 - 180.0


def _ut_hours(timestamp: dt.datetime) -> float:
    return (
        timestamp.hour
        + timestamp.minute / 60.0
        + timestamp.second / 3600.0
        + timestamp.microsecond / 3.6e9
    )


def _local_time_hours(timestamp: dt.datetime, lon_deg: np.ndarray | float) -> np.ndarray:
    return (_ut_hours(timestamp) + np.asarray(lon_deg, dtype=float) / 15.0) % 24.0


def _longitude_for_local_time(timestamp: dt.datetime, local_time_hours: float) -> float:
    return float(_normalize_lon(15.0 * (float(local_time_hours) - _ut_hours(timestamp))))


def _build_sample_points(timestamp: dt.datetime) -> dict[str, tuple[float, float]]:
    points = dict(BASE_STATIONS)
    for latitude in (15.0, 30.0, -15.0, -30.0):
        hemi = "N" if latitude > 0 else "S"
        abs_lat = int(abs(latitude))
        for label, local_time in LOCAL_TIME_SAMPLES.items():
            lon = _longitude_for_local_time(timestamp, local_time)
            points[f"{hemi}{abs_lat}_{label}"] = (latitude, lon)
    return points


def _great_circle_distance_deg(
    lat0_deg: float,
    lon0_deg: float,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
) -> np.ndarray:
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    dlon = lon - lon0
    sin_dlat = np.sin((lat - lat0) / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(lat0) * np.cos(lat) * sin_dlon**2
    a = np.clip(a, 0.0, 1.0)
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(a)))


def _neighborhood_mean(
    values: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    *,
    lat0: float,
    lon0: float,
    radius_deg: float,
) -> dict[str, float]:
    distance_deg = _great_circle_distance_deg(lat0, lon0, lat_grid, lon_grid)
    mask = distance_deg <= float(radius_deg)
    if not np.any(mask):
        return {"mean": float("nan"), "max_abs": float("nan"), "count": 0}
    selected = np.asarray(values, dtype=float)[mask]
    return {
        "mean": float(np.mean(selected)),
        "max_abs": float(np.max(np.abs(selected))),
        "count": int(selected.size),
    }


def _build_ground_evaluators(
    *,
    solution_space: object,
    ionosphere_radius: float,
    point_defs: dict[str, tuple[float, float]],
    proxy_latitudes: Iterable[float],
    proxy_longitude_step: float,
) -> tuple[object, list[str], object, np.ndarray, np.ndarray]:
    point_names = list(point_defs.keys())
    point_lats = np.array([point_defs[name][0] for name in point_names], dtype=float)
    point_lons = np.array([point_defs[name][1] for name in point_names], dtype=float)
    point_ops = build_ground_magnetic_response_operators(
        state_spec=solution_space,
        ground_grid=Grid(lat=point_lats, lon=point_lons),
        ionosphere_radius=ionosphere_radius,
    )

    proxy_lats = []
    proxy_lons = []
    longitudes = np.arange(-180.0, 180.0, float(proxy_longitude_step), dtype=float)
    for lat in proxy_latitudes:
        for lon in longitudes:
            proxy_lats.append(float(lat))
            proxy_lons.append(float(lon))
    proxy_lats_arr = np.array(proxy_lats, dtype=float)
    proxy_lons_arr = np.array(proxy_lons, dtype=float)
    proxy_ops = build_ground_magnetic_response_operators(
        state_spec=solution_space,
        ground_grid=Grid(lat=proxy_lats_arr, lon=proxy_lons_arr),
        ionosphere_radius=ionosphere_radius,
    )
    return point_ops, point_names, proxy_ops, proxy_lats_arr, proxy_lons_arr


def _evaluate_ground_xyz_nT(ops: object, m_ind: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    br = np.asarray(ops.evaluate_radial(m_ind), dtype=float).reshape(-1)
    bh = np.asarray(ops.evaluate_horizontal(m_ind), dtype=float).reshape(2, -1)
    north = -bh[0] * 1e9
    east = bh[1] * 1e9
    down = -br * 1e9
    return north, east, down


def _pack_named_vectors(
    names: list[str],
    north: np.ndarray,
    east: np.ndarray,
    down: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "North_nT": float(north[idx]),
            "East_nT": float(east[idx]),
            "Down_nT": float(down[idx]),
        }
        for idx, name in enumerate(names)
    }


def _sector_mask(local_time_hours: np.ndarray, sector: str) -> np.ndarray:
    lt = np.asarray(local_time_hours, dtype=float) % 24.0
    if sector == "dayside":
        return (lt >= 6.0) & (lt < 18.0)
    if sector == "nightside":
        return ~_sector_mask(lt, "dayside")
    if sector == "noon_sector":
        return (lt >= 9.0) & (lt < 15.0)
    if sector == "dawn_sector":
        return (lt >= 3.0) & (lt < 9.0)
    if sector == "dusk_sector":
        return (lt >= 15.0) & (lt < 21.0)
    if sector == "midnight_sector":
        return (lt >= 21.0) | (lt < 3.0)
    raise ValueError(f"Unknown local-time sector {sector!r}.")


def _proxy_summary(
    *,
    timestamp: dt.datetime,
    proxy_lats: np.ndarray,
    proxy_lons: np.ndarray,
    north_nT: np.ndarray,
) -> dict[str, float]:
    lt = _local_time_hours(timestamp, proxy_lons)
    summary = {
        "global_x_mean_nT": float(np.mean(north_nT)),
        "north_global_x_mean_nT": float(np.mean(north_nT[proxy_lats > 0])),
        "south_global_x_mean_nT": float(np.mean(north_nT[proxy_lats < 0])),
    }
    for sector in (
        "dayside",
        "nightside",
        "noon_sector",
        "dawn_sector",
        "dusk_sector",
        "midnight_sector",
    ):
        mask = _sector_mask(lt, sector)
        if np.any(mask):
            summary[f"{sector}_x_mean_nT"] = float(np.mean(north_nT[mask]))
            north_mask = mask & (proxy_lats > 0)
            south_mask = mask & (proxy_lats < 0)
            summary[f"north_{sector}_x_mean_nT"] = (
                float(np.mean(north_nT[north_mask])) if np.any(north_mask) else float("nan")
            )
            summary[f"south_{sector}_x_mean_nT"] = (
                float(np.mean(north_nT[south_mask])) if np.any(south_mask) else float("nan")
            )
        else:
            summary[f"{sector}_x_mean_nT"] = float("nan")
            summary[f"north_{sector}_x_mean_nT"] = float("nan")
            summary[f"south_{sector}_x_mean_nT"] = float("nan")
    return summary


def _select_one_timestamp_index(raw_time_axis: np.ndarray, time_text: str) -> tuple[int, dt.datetime]:
    window = select_mage_time_window(raw_time_axis, start=time_text, end=time_text)
    if window.indices.size != 1:
        raise ValueError(
            f"Expected exactly one MAGE sample for time {time_text!r}, got {window.indices.size}."
        )
    return int(window.indices[0]), window.timestamps[0]


def _build_case_inputs(
    *,
    fac_model_input: np.ndarray,
    bu: np.ndarray,
    we: np.ndarray,
    wn: np.ndarray,
    case: str,
) -> dict[str, np.ndarray]:
    zeros_fac = np.zeros_like(fac_model_input)
    zeros_br = np.zeros_like(bu)
    zeros_u = np.zeros_like(we)
    if case == "jr_only":
        return {
            "FAC": fac_model_input,
            "Br": zeros_br,
            "u_theta": -zeros_u,
            "u_phi": zeros_u,
        }
    if case == "br_only":
        return {
            "FAC": zeros_fac,
            "Br": bu,
            "u_theta": -zeros_u,
            "u_phi": zeros_u,
        }
    if case == "wind_only":
        return {
            "FAC": zeros_fac,
            "Br": zeros_br,
            "u_theta": -wn,
            "u_phi": we,
        }
    if case == "jr_plus_br":
        return {
            "FAC": fac_model_input,
            "Br": bu,
            "u_theta": -zeros_u,
            "u_phi": zeros_u,
        }
    if case == "all_sources":
        return {
            "FAC": fac_model_input,
            "Br": bu,
            "u_theta": -wn,
            "u_phi": we,
        }
    raise ValueError(f"Unknown case {case!r}.")


def _run_case(
    *,
    run_directory: Path,
    connect_hemispheres: bool,
    timestamp: dt.datetime,
    conductance_hall: np.ndarray,
    conductance_pedersen: np.ndarray,
    fac_model_input: np.ndarray,
    bu: np.ndarray,
    we: np.ndarray,
    wn: np.ndarray,
    ionosphere_lat: np.ndarray,
    ionosphere_lon: np.ndarray,
    magnetosphere_lat: np.ndarray,
    magnetosphere_lon: np.ndarray,
    sqrt_weights_iono_scalar: np.ndarray,
    sqrt_weights_iono_vector: np.ndarray,
    sqrt_weights_mag_geom: np.ndarray,
    point_defs: dict[str, tuple[float, float]],
    times_s: tuple[float, ...],
    case: str,
    proxy_latitudes: tuple[float, ...],
    proxy_longitude_step: float,
) -> dict[str, object]:
    _log(f"Running case={case} connect_hemispheres={connect_hemispheres}")
    dyn = pynamit.Dynamics(
        str(run_directory),
        benchmark_mode=True,
        connect_hemispheres=bool(connect_hemispheres),
    )
    st = dyn.state
    if st.dynamics_mode != DynamicsMode.FULL_INDUCTION:
        raise ValueError(
            f"Expected full_induction benchmark run, got dynamics_mode={st.dynamics_mode!r}."
        )

    point_ops, point_names, proxy_ops, proxy_lats, proxy_lons = _build_ground_evaluators(
        solution_space=st.solution_space,
        ionosphere_radius=st.settings.RI,
        point_defs=point_defs,
        proxy_latitudes=proxy_latitudes,
        proxy_longitude_step=proxy_longitude_step,
    )

    case_inputs = _build_case_inputs(
        fac_model_input=fac_model_input,
        bu=bu,
        we=we,
        wn=wn,
        case=case,
    )

    dyn.set_conductance(
        conductance_hall.reshape(1, -1),
        conductance_pedersen.reshape(1, -1),
        lat=ionosphere_lat,
        lon=ionosphere_lon,
        sqrt_weights=sqrt_weights_iono_scalar,
    )
    dyn.set_FAC(
        case_inputs["FAC"].reshape(1, -1),
        lat=ionosphere_lat,
        lon=ionosphere_lon,
        sqrt_weights=sqrt_weights_iono_scalar,
    )
    dyn.set_Br(
        case_inputs["Br"].reshape(1, -1),
        lat=magnetosphere_lat,
        lon=magnetosphere_lon,
        sqrt_weights=sqrt_weights_mag_geom,
    )
    dyn.set_u(
        case_inputs["u_theta"].reshape(1, -1),
        case_inputs["u_phi"].reshape(1, -1),
        lat=ionosphere_lat,
        lon=ionosphere_lon,
        sqrt_weights=sqrt_weights_iono_vector,
    )

    st.update(dyn.input_manager, dyn.current_time, interpolation=True)
    e_noind, _ = st.calculate_noind_coeffs()

    forcing = np.asarray(st.build_coupled_forcing(e_noind), dtype=float)
    dm_dt0 = np.asarray(forcing[1], dtype=float)
    init_north, init_east, init_down = _evaluate_ground_xyz_nT(point_ops, dm_dt0)
    init_proxy_north, _, _ = _evaluate_ground_xyz_nT(proxy_ops, dm_dt0)

    reduced = st.get_coupled_reduced_time_integration_system(use_dense=bool(st.dense_full_operators))
    n_coeff = int(st.solution_space.index_length)
    y0 = np.zeros((2, n_coeff), dtype=float)
    y0_reduced = reduced.reduce_vector(y0)
    forcing_reduced = reduced.reduce_vector(forcing)
    exp_kwargs = {"max_step_scale": 10.0, "max_substeps": 32768}
    if st.exponential_solver == ExponentialSolverKind.EXPM:
        exp_kwargs["affine_expm_mode"] = "dense"
    elif st.exponential_solver == ExponentialSolverKind.EXPM_MULTIPLY:
        exp_kwargs["affine_expm_mode"] = "action"

    finite_time = {}
    for t_s in times_s:
        y_reduced = st._evolve_linear_state(
            y0_reduced,
            float(t_s),
            linear_operator=reduced.reduced_operator,
            forcing=forcing_reduced,
            exponential_kwargs=exp_kwargs,
        )
        y_full = np.asarray(reduced.expand_vector(y_reduced), dtype=float).reshape(2, n_coeff)
        m_ind = y_full[1]
        north, east, down = _evaluate_ground_xyz_nT(point_ops, m_ind)
        proxy_north, _, _ = _evaluate_ground_xyz_nT(proxy_ops, m_ind)
        finite_time[str(int(round(t_s)))] = {
            "stations": _pack_named_vectors(point_names, north, east, down),
            "proxy_summary": _proxy_summary(
                timestamp=timestamp,
                proxy_lats=proxy_lats,
                proxy_lons=proxy_lons,
                north_nT=proxy_north,
            ),
        }

    return {
        "initial_tendency_nT_per_s": {
            "stations": _pack_named_vectors(point_names, init_north, init_east, init_down),
            "proxy_summary": _proxy_summary(
                timestamp=timestamp,
                proxy_lats=proxy_lats,
                proxy_lons=proxy_lons,
                north_nT=init_proxy_north,
            ),
        },
        "finite_response_nT": finite_time,
    }


def _collect_local_source_means(
    *,
    timestamp: dt.datetime,
    point_defs: dict[str, tuple[float, float]],
    ionosphere_lat: np.ndarray,
    ionosphere_lon: np.ndarray,
    magnetosphere_lat: np.ndarray,
    magnetosphere_lon: np.ndarray,
    fac_up: np.ndarray,
    fac_model_input: np.ndarray,
    jr_converted: np.ndarray,
    bu: np.ndarray,
    we: np.ndarray,
    wn: np.ndarray,
    conductance_hall: np.ndarray,
    conductance_pedersen: np.ndarray,
    radius_deg: float,
) -> dict[str, dict[str, object]]:
    summary = {}
    for name, (lat, lon) in point_defs.items():
        summary[name] = {
            "latitude_deg": float(lat),
            "longitude_deg": float(lon),
            "local_time_hours": float(_local_time_hours(timestamp, lon)),
            "neighborhood_radius_deg": float(radius_deg),
            "raw_fac_upward_A_per_m2": _neighborhood_mean(
                fac_up,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "fac_after_north_flip_A_per_m2": _neighborhood_mean(
                fac_model_input,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "jr_converted_A_per_m2": _neighborhood_mean(
                jr_converted,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "raw_bu_tesla": _neighborhood_mean(
                bu,
                magnetosphere_lat,
                magnetosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "raw_we_m_per_s": _neighborhood_mean(
                we,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "raw_wn_m_per_s": _neighborhood_mean(
                wn,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "hall_conductance_siemens": _neighborhood_mean(
                conductance_hall,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
            "pedersen_conductance_siemens": _neighborhood_mean(
                conductance_pedersen,
                ionosphere_lat,
                ionosphere_lon,
                lat0=lat,
                lon0=lon,
                radius_deg=radius_deg,
            ),
        }
    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5-file",
        default=str(DEFAULT_H5_PATH),
        help="Path to the raw MAGE HDF5 file.",
    )
    parser.add_argument(
        "--run-directory",
        default=str(DEFAULT_RUN_DIRECTORY),
        help="Existing mage_forcing_3 run directory to load in benchmark mode.",
    )
    parser.add_argument(
        "--time",
        default=DEFAULT_TIME,
        help="Exact raw MAGE timestamp to diagnose. Use HH:MM, HH:MM:SS, or ISO datetime.",
    )
    parser.add_argument(
        "--times",
        nargs="*",
        type=float,
        default=list(DEFAULT_TIMES),
        help="Finite response times in seconds. Default: 10 60 120.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["jr_only", "br_only", "wind_only", "jr_plus_br", "all_sources"],
        choices=["jr_only", "br_only", "wind_only", "jr_plus_br", "all_sources"],
        help="Source cases to evaluate.",
    )
    parser.add_argument(
        "--connect-hemispheres",
        nargs="*",
        type=int,
        default=[0, 1],
        choices=[0, 1],
        help="Connect-hemispheres settings to compare. Use 0 and/or 1.",
    )
    parser.add_argument(
        "--proxy-latitudes",
        nargs="*",
        type=float,
        default=list(DEFAULT_LATITUDES),
        help="Low-latitude ring used for X-response proxy summaries.",
    )
    parser.add_argument(
        "--proxy-longitude-step",
        type=float,
        default=DEFAULT_LONGITUDE_STEP,
        help="Longitude spacing in degrees for the low-latitude proxy ring.",
    )
    parser.add_argument(
        "--neighborhood-radius-deg",
        type=float,
        default=DEFAULT_NEIGHBORHOOD_RADIUS_DEG,
        help="Angular radius used for local raw-source summaries.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path. Defaults to stdout only.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    h5_path = Path(args.h5_file).expanduser().resolve()
    run_directory = Path(args.run_directory).expanduser().resolve()
    times_s = tuple(float(value) for value in args.times)
    connect_values = tuple(bool(int(value)) for value in args.connect_hemispheres)

    with h5.File(h5_path, "r") as file:
        source_index, timestamp = _select_one_timestamp_index(file["time"][:], args.time)
        point_defs = _build_sample_points(timestamp)

        fac_raw = np.asarray(file["FAC"][source_index], dtype=float)
        fac_raw = np.nan_to_num(fac_raw, nan=0.0)
        bu = np.asarray(file["Bu"][source_index], dtype=float)
        we = np.asarray(file["We"][source_index], dtype=float)
        wn = np.asarray(file["Wn"][source_index], dtype=float)
        conductance_hall = np.asarray(file["SH"][source_index], dtype=float)
        conductance_pedersen = np.asarray(file["SP"][source_index], dtype=float)
        ionosphere_lat = np.asarray(file["glat"][:], dtype=float)
        ionosphere_lon = np.asarray(file["glon"][:], dtype=float)
        magnetosphere_lat = np.asarray(file["Blat"][:], dtype=float)
        magnetosphere_lon = np.asarray(file["Blon"][:], dtype=float)

    if np.any(conductance_hall <= 0) or np.any(conductance_pedersen <= 0):
        raise ValueError("Selected conductance sample contains non-positive values.")

    fac_model_input = fac_raw.copy()
    fac_model_input[ionosphere_lat > 0] *= -1.0

    sqrt_weights_iono_scalar = compute_spherical_input_sqrt_weights(
        ionosphere_lat,
        ionosphere_lon,
        weighting=IONO_WEIGHTING,
        nmax=NMAX,
    ).reshape(-1)
    sqrt_weights_iono_vector = compute_spherical_input_sqrt_weights(
        ionosphere_lat,
        ionosphere_lon,
        weighting=IONO_WEIGHTING,
        nmax=NMAX,
        vector=True,
    )
    sqrt_weights_mag_geom = compute_spherical_input_sqrt_weights(
        magnetosphere_lat,
        magnetosphere_lon,
        weighting=BR_WEIGHTING,
        periodic_lon=True,
    ).reshape(-1)

    ionosphere_lat_f = ionosphere_lat.reshape(-1)
    ionosphere_lon_f = ionosphere_lon.reshape(-1)
    magnetosphere_lat_f = magnetosphere_lat.reshape(-1)
    magnetosphere_lon_f = magnetosphere_lon.reshape(-1)

    base_dyn = pynamit.Dynamics(str(run_directory), benchmark_mode=True, connect_hemispheres=False)
    b_field = base_dyn.mainfield.discretize(
        Grid(lat=ionosphere_lat_f, lon=ionosphere_lon_f),
        base_dyn.settings.RI,
    )
    radial_factor = np.asarray(b_field.vec.r / b_field.magnitude, dtype=float).reshape(fac_raw.shape)
    jr_converted = fac_model_input * radial_factor

    output = {
        "selected_time": timestamp.isoformat(sep=" "),
        "source_index": int(source_index),
        "paper_anchor": {
            "reference": "Shi et al. 2022 geospace concussion event",
            "sc_time_utc": "2011-10-24 18:31:00",
            "positive_sym_h_impulse_expected": True,
            "heuristic_low_lat_dayside_x_sign": "positive for compressional Br-dominated response",
        },
        "point_definitions": {
            name: {
                "latitude_deg": float(lat),
                "longitude_deg": float(lon),
                "local_time_hours": float(_local_time_hours(timestamp, lon)),
            }
            for name, (lat, lon) in point_defs.items()
        },
        "local_source_means": _collect_local_source_means(
            timestamp=timestamp,
            point_defs=point_defs,
            ionosphere_lat=ionosphere_lat,
            ionosphere_lon=ionosphere_lon,
            magnetosphere_lat=magnetosphere_lat,
            magnetosphere_lon=magnetosphere_lon,
            fac_up=fac_raw,
            fac_model_input=fac_model_input,
            jr_converted=jr_converted,
            bu=bu,
            we=we,
            wn=wn,
            conductance_hall=conductance_hall,
            conductance_pedersen=conductance_pedersen,
            radius_deg=float(args.neighborhood_radius_deg),
        ),
        "responses": {},
    }

    for connect_hemispheres in connect_values:
        connect_key = f"connect_hemispheres_{int(connect_hemispheres)}"
        output["responses"][connect_key] = {}
        for case in args.cases:
            output["responses"][connect_key][case] = _run_case(
                run_directory=run_directory,
                connect_hemispheres=connect_hemispheres,
                timestamp=timestamp,
                conductance_hall=conductance_hall.reshape(-1),
                conductance_pedersen=conductance_pedersen.reshape(-1),
                fac_model_input=fac_model_input.reshape(-1),
                bu=bu.reshape(-1),
                we=we.reshape(-1),
                wn=wn.reshape(-1),
                ionosphere_lat=ionosphere_lat_f,
                ionosphere_lon=ionosphere_lon_f,
                magnetosphere_lat=magnetosphere_lat_f,
                magnetosphere_lon=magnetosphere_lon_f,
                sqrt_weights_iono_scalar=sqrt_weights_iono_scalar,
                sqrt_weights_iono_vector=sqrt_weights_iono_vector,
                sqrt_weights_mag_geom=sqrt_weights_mag_geom,
                point_defs=point_defs,
                times_s=times_s,
                case=case,
                proxy_latitudes=tuple(float(v) for v in args.proxy_latitudes),
                proxy_longitude_step=float(args.proxy_longitude_step),
            )

    json_text = json.dumps(output, indent=2, sort_keys=True)
    print(json_text)
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(json_text + "\n", encoding="utf-8")
        _log(f"Wrote diagnostic JSON to {output_path}")


if __name__ == "__main__":
    main()
