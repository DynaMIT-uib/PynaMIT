"""Adapters for native and bundled external empirical inputs."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Mapping
from functools import cache, lru_cache
from importlib import import_module, resources
from pathlib import Path
from typing import Any

import apexpy
import dipole
import numpy as np

from pynamit.external_inputs.contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    LIBRARY_GEOGRAPHIC_110KM,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    CachedProviderData,
    ExternalInputRequest,
    FallbackCollection,
    InputProviderSpec,
    SampleGrid,
)
from pynamit.geodesy import library_horizontal_to_spherical
from pynamit.geomagnetism import decimal_year

FALLBACK_RESOURCE = resources.files("pynamit.data") / "fallback_inputs.json"
FALLBACK_SCHEMA_VERSION = 7
_INPUT_SOURCE = os.environ.get("PYNAMIT_INPUT_SOURCE", "native").lower()
if _INPUT_SOURCE == "auto":
    _INPUT_SOURCE = "native"

_IONOSPHERE_ALTITUDE_KM = 110.0


def _provider_utc_datetime(date: Any) -> Any:
    """Return a naive UTC datetime for empirical-library calls.

    Historical PynaMIT callers use naive datetimes to mean UTC. Aware
    datetimes are converted to the same convention. This keeps HWM's UTC
    hours consistent with dates passed to Lompe, ApexPy, and pyAMPS.
    """
    if not isinstance(date, dt.datetime) or date.tzinfo is None:
        return date
    return date.astimezone(dt.timezone.utc).replace(tzinfo=None)


@cache
def _load_optional_module(name: str, package: str) -> Any | None:
    """Import and cache one optional input dependency."""
    if _INPUT_SOURCE == "fallback":
        return None

    try:
        if name == package:
            return import_module(package)
        module_name = f"{package}.{name}"
        try:
            return import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            return getattr(import_module(package), name)
    except Exception as exc:
        raise RuntimeError(
            f"Native input source {package!r} is not available. "
            "Install native input dependencies or explicitly call "
            "set_input_source('fallback')."
        ) from exc


def native_inputs_available() -> bool:
    """Return whether all native empirical providers are importable."""
    try:
        try:
            import_module("lompe.conductance")
        except ModuleNotFoundError as exc:
            if exc.name != "lompe.conductance":
                return False
            module = import_module("lompe")
            _ = module.conductance
        import_module("pyamps")
        hwm = import_module("pyhwm2014")
        if not callable(getattr(hwm, "hwm14_vectorized", None)):
            return False
    except Exception:
        return False
    return True


def get_input_source() -> str:
    """Return the preferred input-source mode."""
    return _INPUT_SOURCE


def set_input_source(source: str | None) -> str:
    """Set the preferred input-source mode."""
    global _INPUT_SOURCE

    normalized = "native" if source is None else source.strip().lower()
    if normalized == "auto":
        normalized = "native"
    if normalized not in {"fallback", "native"}:
        raise ValueError("Input source must be 'fallback' or 'native'.")

    _INPUT_SOURCE = normalized
    _load_optional_module.cache_clear()
    os.environ["PYNAMIT_INPUT_SOURCE"] = normalized
    return _INPUT_SOURCE


def _read_fallback(path: os.PathLike[str] | str | None = None) -> FallbackCollection:
    """Read and validate one immutable fallback collection."""
    if path is None:
        with resources.as_file(FALLBACK_RESOURCE) as resource_path:
            collection = FallbackCollection.read(
                resource_path, expected_version=FALLBACK_SCHEMA_VERSION
            )
    else:
        collection = FallbackCollection.read(Path(path), expected_version=FALLBACK_SCHEMA_VERSION)

    if set(collection.providers) != set(PROVIDER_SPECS):
        raise RuntimeError("Fallback provider set differs from the current provider specs.")
    for key, expected in PROVIDER_SPECS.items():
        if collection.providers[key] != expected:
            raise RuntimeError(
                f"Fallback provider specification {key!r} is stale; regenerate the fixture."
            )
    return collection


@lru_cache(maxsize=1)
def _bundled_fallback() -> FallbackCollection:
    """Return the process-local immutable bundled collection."""
    return _read_fallback()


def _load_fallback(path: os.PathLike[str] | str | None = None) -> FallbackCollection:
    """Load immutable fallback data."""
    return _read_fallback(path) if path is not None else _bundled_fallback()


def _load_fallback_snapshot(date: Any) -> FallbackCollection:
    """Return bundled inputs for their declared physical event."""
    event_time = _provider_utc_datetime(date)
    if not isinstance(event_time, dt.datetime):
        raise TypeError("Bundled external inputs require a datetime event time.")

    collection = _load_fallback()
    if collection.event_time is None:
        raise RuntimeError("Bundled external inputs do not declare their event time.")
    bundled_event_time = dt.datetime.fromisoformat(collection.event_time)
    if event_time != bundled_event_time:
        raise ValueError(
            "Bundled external inputs describe only "
            f"{bundled_event_time.isoformat(sep=' ')} UTC; requested "
            f"{event_time.isoformat(sep=' ')} UTC. Use native providers for another event."
        )
    return collection


def _fallback_parameters(
    collection: FallbackCollection, provider_key: str, names: tuple[str, ...]
) -> tuple[Any, ...]:
    """Return physical parameters declared by cached provider data."""
    parameters = collection.conditions.get(provider_key)
    if parameters is None:
        raise RuntimeError(f"Fallback data do not declare {provider_key!r} input conditions.")
    try:
        return tuple(parameters[name] for name in names)
    except KeyError as exc:
        raise RuntimeError(
            f"Fallback data do not declare {provider_key!r} parameter {exc.args[0]!r}."
        ) from exc


def _provider_values(
    values: np.ndarray, *, provider: str, field: str, expected_size: int
) -> np.ndarray:
    """Return flat provider values after validating the sample count."""
    result = np.asarray(values).reshape(-1)
    if result.size != expected_size:
        raise ValueError(
            f"{provider} returned {result.size} {field!r} values for "
            f"a {expected_size}-point request grid."
        )
    return result


def _paired_sample_coordinates(
    lat: np.ndarray, lon: np.ndarray, *, value_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired flat coordinates, expanding 1-D axes."""
    latitude = np.asarray(lat)
    longitude = np.asarray(lon)
    if latitude.shape == longitude.shape and latitude.size == value_size:
        return latitude.reshape(-1), longitude.reshape(-1)
    if latitude.ndim == 1 and longitude.ndim == 1 and latitude.size * longitude.size == value_size:
        lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
        return lat_grid.reshape(-1), lon_grid.reshape(-1)
    raise ValueError(
        "Latitude and longitude must be paired arrays or 1-D axes whose "
        "product matches the provider value count."
    )


def _coerce_request(
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    request: ExternalInputRequest | None,
    *,
    grid_id: str,
) -> ExternalInputRequest:
    """Return a request and validate redundant coordinates."""
    if request is None:
        if lat is None or lon is None:
            raise ValueError(
                "External inputs require source geocentric-GEO lat/lon "
                "or a shared ExternalInputRequest."
            )
        return ExternalInputRequest.from_geocentric_geo(lat, lon, grid_id=grid_id)

    if (lat is None) != (lon is None):
        raise ValueError("lat and lon must be supplied together.")
    if lat is not None:
        supplied = request.source_grid.coordinate_convention.coordinate_identity(lat, lon)
        if supplied != request.source_grid.coordinate_identity:
            raise ValueError("Explicit coordinates do not match the shared request source grid.")
    return request


def save_fallback_dataset(
    destination: os.PathLike[str] | str,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    pedersen: np.ndarray,
    hall: np.ndarray,
    jr: np.ndarray,
    u_theta: np.ndarray,
    u_phi: np.ndarray,
    time: np.ndarray | None = None,
    grid_id: str = "default",
    wind_lat: np.ndarray | None = None,
    wind_lon: np.ndarray | None = None,
    event_time: Any | None = None,
    indent: int | None = 2,
) -> Path:
    """Save a compact typed fallback collection for tests/examples."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    hall = np.asarray(hall).reshape(-1)
    pedersen = np.asarray(pedersen).reshape(-1)
    jr = np.asarray(jr).reshape(-1)
    if not (hall.size == pedersen.size == jr.size):
        raise ValueError("Hall, Pedersen, and jr arrays must have equal sizes.")
    scalar_lat, scalar_lon = _paired_sample_coordinates(lat, lon, value_size=hall.size)
    scalar_request = ExternalInputRequest.from_geocentric_geo(
        scalar_lat, scalar_lon, grid_id=str(grid_id), sampling_geometry={"type": "sample_points"}
    )

    u_theta = np.asarray(u_theta).reshape(-1)
    u_phi = np.asarray(u_phi).reshape(-1)
    if u_theta.size != u_phi.size:
        raise ValueError("Wind component arrays must have equal sizes.")
    if wind_lat is None:
        wind_lat = lat
    if wind_lon is None:
        wind_lon = lon
    wind_lat, wind_lon = _paired_sample_coordinates(wind_lat, wind_lon, value_size=u_theta.size)
    wind_source_id = (
        str(grid_id)
        if scalar_lat.size == wind_lat.size
        and scalar_request.source_grid.coordinate_convention.coordinate_identity(
            wind_lat, wind_lon
        )
        == scalar_request.source_grid.coordinate_identity
        else f"{grid_id}-wind"
    )
    wind_request = (
        scalar_request
        if wind_source_id == str(grid_id)
        else ExternalInputRequest.from_geocentric_geo(
            wind_lat, wind_lon, grid_id=wind_source_id, sampling_geometry={"type": "sample_points"}
        )
    )

    grids: dict[str, SampleGrid] = {}
    for request_object in (scalar_request, wind_request):
        source = request_object.source_grid
        provider_grid = request_object.grid_for(LIBRARY_GEOGRAPHIC_110KM)
        grids[source.grid_id] = source
        grids[provider_grid.grid_id] = provider_grid

    datasets = {
        CONDUCTANCE_PROVIDER_SPEC.key: {
            scalar_request.source_grid.grid_id: CachedProviderData(
                spec=CONDUCTANCE_PROVIDER_SPEC,
                source_grid=scalar_request.source_grid,
                request_grid=scalar_request.grid_for(CONDUCTANCE_PROVIDER_SPEC),
                values={"hall": hall, "pedersen": pedersen},
            )
        },
        BOUNDARY_JR_PROVIDER_SPEC.key: {
            scalar_request.source_grid.grid_id: CachedProviderData(
                spec=BOUNDARY_JR_PROVIDER_SPEC,
                source_grid=scalar_request.source_grid,
                request_grid=scalar_request.grid_for(BOUNDARY_JR_PROVIDER_SPEC),
                values={"jr": jr},
            )
        },
        NEUTRAL_WIND_PROVIDER_SPEC.key: {
            wind_request.source_grid.grid_id: CachedProviderData(
                spec=NEUTRAL_WIND_PROVIDER_SPEC,
                source_grid=wind_request.source_grid,
                request_grid=wind_request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC),
                values={"u_theta": u_theta, "u_phi": u_phi},
            )
        },
    }
    event_time_value = (
        None
        if event_time is None
        else event_time.isoformat()
        if hasattr(event_time, "isoformat")
        else str(event_time)
    )
    collection = FallbackCollection(
        version=FALLBACK_SCHEMA_VERSION,
        event_time=event_time_value,
        time=np.asarray([0.0] if time is None else time),
        grids=grids,
        providers=PROVIDER_SPECS,
        datasets=datasets,
    )
    collection.write(destination_path, indent=indent)
    return destination_path


def _dataset_description(dataset: CachedProviderData) -> str:
    """Return a compact physical-grid description."""
    geometry = dataset.source_grid.sampling_geometry
    provenance = dataset.source_grid.provenance
    origin = provenance.get("originating_model_frame", {})
    details = [str(origin.get("horizontal_coordinate_system", "runtime"))]
    if "ncs" in geometry:
        details.append(f"Ncs={geometry['ncs']}")
    if "epoch" in origin:
        details.append(f"epoch={float(origin['epoch']):.6f}")
    details.append(dataset.source_grid.coordinate_convention.coordinate_system)
    details.append(f"{dataset.source_grid.size} ordered points")
    return f"{dataset.source_grid.grid_id} ({', '.join(details)})"


def _dataset_sort_key(item: tuple[str, CachedProviderData]) -> tuple[Any, ...]:
    """Return a natural diagnostic ordering for cached source grids."""
    source_grid_id, dataset = item
    geometry = dataset.source_grid.sampling_geometry
    origin = dataset.source_grid.provenance.get("originating_model_frame", {})
    horizontal = str(origin.get("horizontal_coordinate_system", ""))
    geometry_order = {"geocentric_geographic": 0, "centered_dipole": 1}.get(horizontal, 2)
    try:
        ncs = int(geometry.get("ncs"))
    except (TypeError, ValueError):
        ncs = 10**9
    try:
        epoch = float(origin.get("epoch"))
    except (TypeError, ValueError):
        epoch = float("inf")
    return geometry_order, ncs, epoch, str(source_grid_id)


def _select_fallback_entry(
    entries: Mapping[str, CachedProviderData],
    request: ExternalInputRequest,
    quantity: str,
    *,
    spec: InputProviderSpec,
) -> CachedProviderData:
    """Select an exact source/request-grid pair for one provider."""
    if not entries:
        raise ValueError(f"No fallback {quantity} data available.")

    expected_request_grid = request.grid_for(spec)
    compatible = []
    for source_grid_id, dataset in entries.items():
        if dataset.spec != spec:
            continue
        compatible.append((source_grid_id, dataset))
        if dataset.source_grid.coordinate_identity == request.source_grid.coordinate_identity:
            if (
                dataset.request_grid.coordinate_identity
                != expected_request_grid.coordinate_identity
            ):
                raise RuntimeError(f"Fallback {quantity} provider-request grid is stale.")
            return dataset

    if not compatible:
        raise ValueError(
            f"No fallback {quantity} dataset uses provider specification "
            f"{spec.key!r} ({spec.signature})."
        )

    available = "; ".join(
        _dataset_description(dataset) for _, dataset in sorted(compatible, key=_dataset_sort_key)
    )
    raise ValueError(
        f"No fallback {quantity} data matches the requested ordered "
        f"{request.source_grid.size}-point source grid. "
        f"Available compatible grids: {available}"
    )


def _validated_centered_dipole(request: ExternalInputRequest) -> dipole.Dipole:
    """Return the request dipole after checking its paired GEO view."""
    model_epoch = request.model_epoch
    if model_epoch is None:
        raise ValueError("A centered-dipole external-input view requires a model epoch.")

    model_grid = request.model_grid
    source_grid = request.source_grid
    centered_dipole = dipole.Dipole(model_epoch)
    geographic_lat, geographic_lon = centered_dipole.mag2geo(model_grid.lat, model_grid.lon)
    longitude_error = (np.asarray(geographic_lon) - source_grid.lon + 180.0) % 360.0 - 180.0
    if not np.allclose(geographic_lat, source_grid.lat, rtol=0.0, atol=1e-8) or not np.allclose(
        longitude_error, 0.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError(
            "Centered-dipole model coordinates, model_epoch, and physical GEO samples disagree."
        )
    return centered_dipole


def _hardy_euv_from_coordinate_views(
    conductance,
    request: ExternalInputRequest,
    provider_date: dt.datetime,
    centered_dipole: dipole.Dipole | None,
    *,
    kp: int,
    starlight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Hardy magnetically and EUV in physical GEO."""
    model_grid = request.model_grid
    source_grid = request.source_grid
    if centered_dipole is None:
        provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
        apex = apexpy.Apex(date=provider_date, refh=_IONOSPHERE_ALTITUDE_KM)
        auroral_lat, auroral_lon = apex.geo2apex(
            provider_grid.lat, provider_grid.lon, _IONOSPHERE_ALTITUDE_KM
        )
        centered_dipole = dipole.Dipole(decimal_year(provider_date))
    else:
        auroral_lat, auroral_lon = model_grid.lat, model_grid.lon

    mlt = centered_dipole.mlon2mlt(auroral_lon, provider_date)
    solar_zenith_angle = conductance.sunlight.sza(source_grid.lat, source_grid.lon, provider_date)
    euv_hall, euv_pedersen = conductance.EUV_conductance(
        solar_zenith_angle, 100, "hp", calibration="MoenBrekke1993"
    )
    auroral_hall, auroral_pedersen = conductance.hardy(auroral_lat, mlt, int(kp), "hp")
    background_squared = float(starlight) ** 2
    return (
        np.sqrt(auroral_hall**2 + euv_hall**2 + background_squared),
        np.sqrt(auroral_pedersen**2 + euv_pedersen**2 + background_squared),
    )


def get_conductance_inputs(
    date: Any,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    *,
    request: ExternalInputRequest | None = None,
    kp: int,
    starlight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Pedersen/Hall conductance on the source PynaMIT grid.

    ``kp`` is the Hardy activity index and ``starlight`` is the
    nightside background conductance in siemens.

    Returns
    -------
    pedersen, hall, lat, lon : ndarray
        Conductance in siemens followed by the source-grid coordinates.
        Lompe's native Hardy/EUV routine returns Hall first; that order
        is converted explicitly at this adapter boundary.
    """
    request = _coerce_request(lat, lon, request, grid_id="runtime-conductance-source")
    source_grid = request.source_grid
    centered_dipole_convention = CONDUCTANCE_PROVIDER_SPEC.request_coordinate_views["model"]
    centered_dipole = (
        _validated_centered_dipole(request)
        if request.model_grid.coordinate_convention == centered_dipole_convention
        else None
    )
    conductance = _load_optional_module("conductance", "lompe")

    if conductance is not None:
        provider_date = _provider_utc_datetime(date)
        hall, pedersen = _hardy_euv_from_coordinate_views(
            conductance,
            request,
            provider_date,
            centered_dipole,
            kp=int(kp),
            starlight=float(starlight),
        )
        hall = _provider_values(
            hall, provider="Lompe Hardy/EUV", field="hall", expected_size=source_grid.size
        )
        pedersen = _provider_values(
            pedersen, provider="Lompe Hardy/EUV", field="pedersen", expected_size=source_grid.size
        )
        return (
            pedersen,
            hall,
            np.array(source_grid.lat, copy=True),
            np.array(source_grid.lon, copy=True),
        )

    if get_input_source() == "fallback":
        collection = _load_fallback_snapshot(date)
        parameter_names = ("kp", "starlight")
        bundled_parameters = _fallback_parameters(
            collection, CONDUCTANCE_PROVIDER_SPEC.key, parameter_names
        )
        requested_parameters = (float(kp), float(starlight))
        if not np.allclose(requested_parameters, bundled_parameters):
            conditions = dict(zip(parameter_names, bundled_parameters, strict=True))
            raise ValueError(
                f"Bundled conductance fallback data use {conditions}; "
                "different provider parameters require native inputs."
            )
        dataset = _select_fallback_entry(
            collection.datasets[CONDUCTANCE_PROVIDER_SPEC.key],
            request,
            "conductance",
            spec=collection.providers[CONDUCTANCE_PROVIDER_SPEC.key],
        )
        return (
            np.array(dataset.values["pedersen"], copy=True),
            np.array(dataset.values["hall"], copy=True),
            np.array(dataset.source_grid.lat, copy=True),
            np.array(dataset.source_grid.lon, copy=True),
        )

    raise RuntimeError("Native conductance inputs are not available.")


def get_boundary_jr_inputs(
    date: Any,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    *,
    request: ExternalInputRequest | None = None,
    v: float,
    By: float,
    Bz: float,
    tilt: float,
    f107: float,
    minlat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AMPS upward radial current on the source PynaMIT grid.

    The AMPS drivers are solar-wind speed ``v`` in km/s, IMF ``By`` and
    ``Bz`` in nT, dipole ``tilt`` in degrees, F10.7 flux ``f107`` in
    sfu, and the model cutoff ``minlat`` in degrees.
    """
    request = _coerce_request(lat, lon, request, grid_id="runtime-boundary-jr-source")
    source_grid = request.source_grid
    centered_dipole_convention = BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_views["model"]
    centered_dipole = (
        _validated_centered_dipole(request)
        if request.model_grid.coordinate_convention == centered_dipole_convention
        else None
    )
    pyamps = _load_optional_module("pyamps", "pyamps")

    if pyamps is not None:
        provider_date = _provider_utc_datetime(date)
        if centered_dipole is None:
            provider_grid = request.grid_for(BOUNDARY_JR_PROVIDER_SPEC)
            apex = apexpy.Apex(date=provider_date, refh=_IONOSPHERE_ALTITUDE_KM)
            mlat, mlon = apex.geo2apex(
                provider_grid.lat, provider_grid.lon, _IONOSPHERE_ALTITUDE_KM
            )
            mlt = pyamps.mlon_to_mlt(mlon, provider_date, decimal_year(provider_date))
        else:
            mlat, mlon = request.model_grid.lat, request.model_grid.lon
            mlt = centered_dipole.mlon2mlt(mlon, provider_date)
        coeff_path = os.path.join(
            os.path.dirname(pyamps.__file__),
            "coefficients",
            "SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0104.txt",
        )
        amps = pyamps.AMPS(
            float(v),
            float(By),
            float(Bz),
            float(tilt),
            float(f107),
            minlat=float(minlat),
            coeff_fn=coeff_path,
        )
        jr = (
            _provider_values(
                amps.get_upward_current(mlat=mlat, mlt=mlt),
                provider="pyAMPS",
                field="jr",
                expected_size=source_grid.size,
            )
            * 1e-6
        )
        jr[np.abs(mlat) < float(minlat)] = 0
        return (jr, np.array(source_grid.lat, copy=True), np.array(source_grid.lon, copy=True))

    if get_input_source() == "fallback":
        collection = _load_fallback_snapshot(date)
        parameter_names = ("v", "By", "Bz", "tilt", "f107", "minlat")
        bundled_parameters = _fallback_parameters(
            collection, BOUNDARY_JR_PROVIDER_SPEC.key, parameter_names
        )
        requested_parameters = tuple(float(value) for value in (v, By, Bz, tilt, f107, minlat))
        if not np.allclose(requested_parameters, bundled_parameters):
            conditions = dict(zip(parameter_names, bundled_parameters, strict=True))
            raise ValueError(
                f"Bundled AMPS fallback data use {conditions}; "
                "different provider parameters require native inputs."
            )
        dataset = _select_fallback_entry(
            collection.datasets[BOUNDARY_JR_PROVIDER_SPEC.key],
            request,
            "boundary_jr",
            spec=collection.providers[BOUNDARY_JR_PROVIDER_SPEC.key],
        )
        return (
            np.array(dataset.values["jr"], copy=True),
            np.array(dataset.source_grid.lat, copy=True),
            np.array(dataset.source_grid.lon, copy=True),
        )

    raise RuntimeError("Native FAC/jr inputs are not available.")


def _hwm_iyd(date: Any) -> int:
    """Return HWM's YYDDD integer date code."""
    return int(date.year % 100) * 1000 + int(date.timetuple().tm_yday)


def _hwm_utc_hours(date: Any) -> float:
    """Return UTC hours including seconds and microseconds."""
    return float(date.hour + date.minute / 60.0 + date.second / 3600.0 + date.microsecond / 3.6e9)


def _library_horizontal_wind_to_spherical(
    request: ExternalInputRequest, zonal_east: np.ndarray, meridional_north: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the spherical east/north identity used by all adapters."""
    source_grid = request.source_grid
    provider_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    zonal_east, meridional_north = np.broadcast_arrays(
        np.asarray(zonal_east, dtype=float), np.asarray(meridional_north, dtype=float)
    )
    if zonal_east.size != source_grid.size:
        raise ValueError("HWM wind values must match the request grid size.")
    if not np.array_equal(provider_grid.lat, source_grid.lat):
        raise RuntimeError("Library request latitude must equal source latitude.")
    if not np.array_equal(provider_grid.lon, source_grid.lon):
        raise RuntimeError("Library request longitude must equal source longitude.")
    u_theta, u_phi = library_horizontal_to_spherical(
        zonal_east.reshape(-1), meridional_north.reshape(-1)
    )
    return u_theta, u_phi


def get_wind_inputs(
    date: Any,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    *,
    request: ExternalInputRequest | None = None,
    ap: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return HWM wind on the shared empirical-input source grid.

    ``ap`` is the two-value geomagnetic-activity input expected by the
    vectorized pyHWM2014 interface.
    """
    request = _coerce_request(lat, lon, request, grid_id="runtime-neutral-wind-source")
    source_grid = request.source_grid
    provider_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    ap = tuple(float(value) for value in ap)
    if len(ap) != 2:
        raise ValueError("HWM ap must contain the two values expected by pyHWM2014.")
    pyhwm2014 = _load_optional_module("pyhwm2014", "pyhwm2014")

    if pyhwm2014 is not None:
        provider_date = _provider_utc_datetime(date)
        evaluator = getattr(pyhwm2014, "hwm14_vectorized", None)
        if not callable(evaluator):
            raise RuntimeError(
                "pyhwm2014.hwm14_vectorized is required; update the "
                "pyHWM14 main-branch dependency."
            )
        zonal_east, meridional_north = evaluator(
            alt_km=np.full(provider_grid.size, _IONOSPHERE_ALTITUDE_KM),
            glat_deg=provider_grid.lat,
            glon_deg=provider_grid.lon,
            utc_hours=np.full(provider_grid.size, _hwm_utc_hours(provider_date)),
            iyd=_hwm_iyd(provider_date),
            ap=list(ap),
        )
        u_theta, u_phi = _library_horizontal_wind_to_spherical(
            request, zonal_east, meridional_north
        )
        return (
            u_theta,
            u_phi,
            np.array(source_grid.lat, copy=True),
            np.array(source_grid.lon, copy=True),
            None,
        )

    if get_input_source() != "fallback":
        raise RuntimeError("Native neutral-wind inputs are not available.")

    collection = _load_fallback_snapshot(date)
    (bundled_ap,) = _fallback_parameters(collection, NEUTRAL_WIND_PROVIDER_SPEC.key, ("ap",))
    if not np.allclose(ap, bundled_ap):
        raise ValueError(
            f"Bundled HWM fallback data use ap={tuple(bundled_ap)}; "
            "different provider parameters require native inputs."
        )

    dataset = _select_fallback_entry(
        collection.datasets[NEUTRAL_WIND_PROVIDER_SPEC.key],
        request,
        "neutral_wind",
        spec=collection.providers[NEUTRAL_WIND_PROVIDER_SPEC.key],
    )
    return (
        np.array(dataset.values["u_theta"], copy=True),
        np.array(dataset.values["u_phi"], copy=True),
        np.array(dataset.source_grid.lat, copy=True),
        np.array(dataset.source_grid.lon, copy=True),
        None,
    )
