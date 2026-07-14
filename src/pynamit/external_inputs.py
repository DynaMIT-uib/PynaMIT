"""
Helpers for loading conductance, currents, and winds.

This module encapsulates optional dependencies (lompe, pyamps,
pyhwm2014). Bundled sample data are available for validation tests and
small examples, but must be requested explicitly with
``PYNAMIT_INPUT_SOURCE=fallback`` or ``set_input_source("fallback")``.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import cache, lru_cache
from importlib import import_module, resources
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.sphere.grid import Grid

FALLBACK_RESOURCE = resources.files("pynamit.data") / "fallback_inputs.json"
_INPUT_SOURCE = os.environ.get("PYNAMIT_INPUT_SOURCE", "native").lower()
if _INPUT_SOURCE == "auto":
    _INPUT_SOURCE = "native"


@cache
def _load_optional_module(name: str, package: str) -> Any | None:
    """Import and cache one optional input dependency."""
    if _INPUT_SOURCE == "fallback":
        return None

    try:
        if name == package:
            return import_module(package)
        try:
            return import_module(f"{package}.{name}")
        except ModuleNotFoundError:
            return getattr(import_module(package), name)
    except Exception as exc:
        raise RuntimeError(
            f"Native input source {package!r} is not available. "
            "Install the native input dependencies or explicitly call "
            "set_input_source('fallback') to use bundled sample data."
        ) from exc


def native_inputs_available() -> bool:
    """Return True if lompe, pyamps, and pyhwm2014 can be imported."""
    try:
        try:
            import_module("lompe.conductance")
        except ModuleNotFoundError:
            module = import_module("lompe")
            _ = module.conductance
        import_module("pyamps")
        import_module("pyhwm2014")
    except Exception:
        return False
    return True


def get_input_source() -> str:
    """Return the preferred input source mode."""
    return _INPUT_SOURCE


def set_input_source(source: str | None) -> str:
    """Set the preferred input source mode."""
    global _INPUT_SOURCE

    if source is None:
        source = "native"

    normalized = source.strip().lower()
    if normalized == "auto":
        normalized = "native"
    if normalized not in {"fallback", "native"}:
        raise ValueError("Input source must be 'fallback' or 'native'.")

    _INPUT_SOURCE = normalized
    _load_optional_module.cache_clear()

    os.environ["PYNAMIT_INPUT_SOURCE"] = normalized

    return _INPUT_SOURCE


def _read_fallback(path: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    """Read and normalize one fallback-input dataset."""
    if path is None:
        with resources.as_file(FALLBACK_RESOURCE) as resource_path:
            payload = json.loads(resource_path.read_text())
    else:
        payload = json.loads(Path(path).read_text())

    if "version" not in payload:
        # Legacy format: promote to version 2 with a single grid entry.
        lat = np.asarray(payload["lat"])
        lon = np.asarray(payload["lon"])
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        flattened_lat = lat_grid.reshape(-1)
        flattened_lon = lon_grid.reshape(-1)
        payload = {
            "version": 2,
            "time": payload.get("time", [0.0]),
            "wind": {
                "lat": flattened_lat.tolist(),
                "lon": flattened_lon.tolist(),
                "u_theta": np.asarray(payload["u_theta"]).reshape(-1).tolist(),
                "u_phi": np.asarray(payload["u_phi"]).reshape(-1).tolist(),
            },
            "conductance": {
                str(flattened_lat.size): {
                    "lat": flattened_lat.tolist(),
                    "lon": flattened_lon.tolist(),
                    "hall": np.asarray(payload["hall"]).reshape(-1).tolist(),
                    "pedersen": np.asarray(payload["pedersen"]).reshape(-1).tolist(),
                }
            },
            "jr": {
                str(flattened_lat.size): {
                    "lat": flattened_lat.tolist(),
                    "lon": flattened_lon.tolist(),
                    "jr": np.asarray(payload["jr"]).reshape(-1).tolist(),
                }
            },
        }

    fallback: dict[str, Any] = {
        "version": int(payload.get("version", 2)),
        "time": np.asarray(payload.get("time", [0.0])),
        "wind": {
            "lat": np.asarray(payload["wind"]["lat"]),
            "lon": np.asarray(payload["wind"]["lon"]),
            "u_theta": np.asarray(payload["wind"]["u_theta"]),
            "u_phi": np.asarray(payload["wind"]["u_phi"]),
        },
        "conductance": {},
        "jr": {},
    }

    for key, entry in payload.get("conductance", {}).items():
        fallback["conductance"][str(key)] = {
            "lat": np.asarray(entry["lat"]),
            "lon": np.asarray(entry["lon"]),
            "hall": np.asarray(entry["hall"]),
            "pedersen": np.asarray(entry["pedersen"]),
        }

    for key, entry in payload.get("jr", {}).items():
        fallback["jr"][str(key)] = {
            "lat": np.asarray(entry["lat"]),
            "lon": np.asarray(entry["lon"]),
            "jr": np.asarray(entry["jr"]),
        }

    return fallback


@lru_cache(maxsize=1)
def _bundled_fallback() -> dict[str, Any]:
    """Return the process-local parsed bundled fallback data."""
    return _read_fallback()


def _load_fallback(path: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    """Load fallback data, returning values owned by the caller."""
    if path is not None:
        return _read_fallback(path)
    return deepcopy(_bundled_fallback())


def _expand_time_series(data: np.ndarray, time: np.ndarray | None) -> np.ndarray:
    base = np.asarray(data).reshape(-1)
    if time is None or time.size <= 1:
        return base
    scaling = np.linspace(1.0, 2.0, time.size)[:, None]
    return scaling * base[None, :]


def save_fallback_dataset(
    destination: os.PathLike[str] | str,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    hall: np.ndarray,
    pedersen: np.ndarray,
    jr: np.ndarray,
    u_theta: np.ndarray,
    u_phi: np.ndarray,
    time: np.ndarray | None = None,
    grid_id: str = "default",
    wind_lat: np.ndarray | None = None,
    wind_lon: np.ndarray | None = None,
    indent: int | None = 2,
) -> Path:
    """Save a fallback input dataset to a JSON file."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if wind_lat is None:
        wind_lat = lat
    if wind_lon is None:
        wind_lon = lon

    lat = np.asarray(lat).reshape(-1)
    lon = np.asarray(lon).reshape(-1)
    wind_lat = np.asarray(wind_lat).reshape(-1)
    wind_lon = np.asarray(wind_lon).reshape(-1)

    payload = {
        "version": 2,
        "time": np.asarray([0.0] if time is None else np.asarray(time).reshape(-1)).tolist(),
        "wind": {
            "lat": wind_lat.tolist(),
            "lon": wind_lon.tolist(),
            "u_theta": np.asarray(u_theta).reshape(-1).tolist(),
            "u_phi": np.asarray(u_phi).reshape(-1).tolist(),
        },
        "conductance": {
            grid_id: {
                "lat": lat.tolist(),
                "lon": lon.tolist(),
                "hall": np.asarray(hall).reshape(-1).tolist(),
                "pedersen": np.asarray(pedersen).reshape(-1).tolist(),
            }
        },
        "jr": {
            grid_id: {
                "lat": lat.tolist(),
                "lon": lon.tolist(),
                "jr": np.asarray(jr).reshape(-1).tolist(),
            }
        },
    }

    destination_path.write_text(
        json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
    )
    return destination_path


def _select_fallback_entry(
    entries: dict[str, dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray, quantity: str
) -> dict[str, np.ndarray]:
    if not entries:
        raise ValueError(f"No fallback {quantity} data available.")

    def _entry_sort_key(value: str) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    if lat.size == 0 or lon.size == 0:
        key = sorted(entries.keys(), key=_entry_sort_key)[0]
        return entries[key]

    target_hash = Grid(lat=lat, lon=lon).hash
    for entry in entries.values():
        entry_lat = np.asarray(entry["lat"])
        entry_lon = np.asarray(entry["lon"])
        if entry_lat.size != lat.size or entry_lon.size != lon.size:
            continue
        if Grid(lat=entry_lat, lon=entry_lon).hash == target_hash:
            return entry

    available = ", ".join(sorted(entries.keys()))
    raise ValueError(
        f"No fallback {quantity} data matches a grid with {lat.size} points. "
        f"Available grid sizes: {available}"
    )


def get_conductance_inputs(
    date: Any, lat: np.ndarray, lon: np.ndarray, time: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Hall and Pedersen conductance on grid."""
    source = get_input_source()
    conductance = _load_optional_module("conductance", "lompe")
    if conductance is not None:
        hall, pedersen = conductance.hardy_EUV(lon, lat, 5, date, starlight=1, dipole=True)
        hall = _expand_time_series(hall, time)
        pedersen = _expand_time_series(pedersen, time)
        return hall, pedersen, lat, lon

    if source == "fallback":
        fallback = _load_fallback()
        entry = _select_fallback_entry(fallback["conductance"], lat, lon, "conductance")
        hall = _expand_time_series(entry["hall"], time)
        pedersen = _expand_time_series(entry["pedersen"], time)
        return hall, pedersen, entry["lat"], entry["lon"]

    raise RuntimeError("Native conductance inputs are not available.")


def get_jr_inputs(
    date: Any, lat: np.ndarray, lon: np.ndarray, time: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upward current density on the requested grid."""
    source = get_input_source()
    dipole = _load_optional_module("dipole", "dipole")
    pyamps = _load_optional_module("pyamps", "pyamps")

    if dipole is not None and pyamps is not None:
        d = dipole.Dipole(date.year)
        coeff_path = os.path.join(
            os.path.dirname(pyamps.__file__),
            "coefficients",
            "SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0104.txt",
        )
        amps = pyamps.AMPS(300, 0, -4, 20, 100, minlat=50, coeff_fn=coeff_path)
        mlt = d.mlon2mlt(lon, date)
        jr = amps.get_upward_current(mlat=lat, mlt=mlt) * 1e-6
        jr[np.abs(lat) < 50] = 0
        jr = _expand_time_series(jr, time)
        return jr, lat, lon

    if source == "fallback":
        fallback = _load_fallback()
        entry = _select_fallback_entry(fallback["jr"], lat, lon, "jr")
        jr = _expand_time_series(entry["jr"], time)
        return jr, entry["lat"], entry["lon"]

    raise RuntimeError("Native FAC/jr inputs are not available.")


def get_wind_inputs(
    date: Any, use_wind: bool, time: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Return neutral wind components if requested."""
    if not use_wind:
        return None

    source = get_input_source()
    pyhwm2014 = _load_optional_module("pyhwm2014", "pyhwm2014")
    if pyhwm2014 is not None:
        model = pyhwm2014.HWM142D(
            alt=110.0,
            ap=[35, 35],
            glatlim=[-88.5, 88.5],
            glatstp=1.5,
            glonlim=[-180.0, 180.0],
            glonstp=3.0,
            option=6,
            verbose=False,
            ut=date.hour + date.minute / 60.0,
            day=date.timetuple().tm_yday,
        )
        u_theta = -model.Vwind.reshape(-1)
        u_phi = model.Uwind.reshape(-1)
        lat_grid, lon_grid = np.meshgrid(model.glatbins, model.glonbins, indexing="ij")
        u_theta = _expand_time_series(u_theta, time)
        u_phi = _expand_time_series(u_phi, time)
        weights = np.sqrt(np.sin(np.deg2rad(90.0 - lat_grid.reshape(-1))))
        weights = np.tile(weights, (2, 1))
        return u_theta, u_phi, lat_grid.reshape(-1), lon_grid.reshape(-1), weights

    if source != "fallback":
        raise RuntimeError("Native neutral-wind inputs are not available.")

    fallback = _load_fallback()
    wind_data = fallback["wind"]
    lat_grid = wind_data["lat"]
    lon_grid = wind_data["lon"]
    u_theta = _expand_time_series(wind_data["u_theta"], time)
    u_phi = _expand_time_series(wind_data["u_phi"], time)
    weights = np.sqrt(np.sin(np.deg2rad(90.0 - lat_grid)))
    weights = np.tile(weights, (2, 1))
    return u_theta, u_phi, lat_grid, lon_grid, weights
