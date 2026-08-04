"""Adapters for native and bundled external empirical inputs."""

from __future__ import annotations

import os
from functools import cache, lru_cache
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Mapping

import apexpy
import numpy as np

from pynamit.geodesy import library_horizontal_to_spherical

from pynamit.external_input_contracts import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    ExternalInputRequest,
    FallbackCollection,
    ProviderDataset,
    ProviderSpec,
    SampleGrid,
    LIBRARY_GEOGRAPHIC_110KM,
)
from pynamit.geomagnetism import decimal_year

FALLBACK_RESOURCE = resources.files("pynamit.data") / "fallback_inputs.json"
FALLBACK_SCHEMA_VERSION = 4
_INPUT_SOURCE = os.environ.get("PYNAMIT_INPUT_SOURCE", "native").lower()
if _INPUT_SOURCE == "auto":
    _INPUT_SOURCE = "native"

_HWM_ALTITUDE_KM = 110.0
_HWM_AP = (-1, 35)


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
            "Install native input dependencies or explicitly call "
            "set_input_source('fallback')."
        ) from exc


def native_inputs_available() -> bool:
    """Return whether all native empirical providers are importable."""
    try:
        try:
            import_module("lompe.conductance")
        except ModuleNotFoundError:
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


def _read_fallback(
    path: os.PathLike[str] | str | None = None,
) -> FallbackCollection:
    """Read and validate one immutable fallback collection."""
    if path is None:
        with resources.as_file(FALLBACK_RESOURCE) as resource_path:
            collection = FallbackCollection.read(
                resource_path,
                expected_version=FALLBACK_SCHEMA_VERSION,
            )
    else:
        collection = FallbackCollection.read(
            Path(path),
            expected_version=FALLBACK_SCHEMA_VERSION,
        )

    if set(collection.providers) != set(PROVIDER_SPECS):
        raise RuntimeError(
            "Fallback provider set differs from the current adapter registry."
        )
    for key, expected in PROVIDER_SPECS.items():
        if collection.providers[key] != expected:
            raise RuntimeError(
                f"Fallback provider specification {key!r} is stale; "
                "regenerate the fixture."
            )
    return collection


@lru_cache(maxsize=1)
def _bundled_fallback() -> FallbackCollection:
    """Return the process-local immutable bundled collection."""
    return _read_fallback()


def _load_fallback(
    path: os.PathLike[str] | str | None = None,
) -> FallbackCollection:
    """Load immutable fallback data."""
    return _read_fallback(path) if path is not None else _bundled_fallback()


def _expand_time_series(
    data: np.ndarray,
    time: np.ndarray | None,
) -> np.ndarray:
    """Construct deliberately synthetic multi-time demonstration values.

    Native providers are evaluated once at the base event time. When the
    test/example ``multi_data`` path requests several times, values are
    scaled from one to two solely to exercise multi-step storage,
    interpolation, and evolution logic. This is not a physical forecast.
    """
    base = np.array(data, copy=True).reshape(-1)
    if time is None or np.asarray(time).size <= 1:
        return base
    scaling = np.linspace(1.0, 2.0, np.asarray(time).size)[:, None]
    return scaling * base[None, :]


def _paired_sample_coordinates(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    value_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired flat coordinates, expanding 1-D axes when necessary."""
    latitude = np.asarray(lat)
    longitude = np.asarray(lon)
    if latitude.shape == longitude.shape and latitude.size == value_size:
        return latitude.reshape(-1), longitude.reshape(-1)
    if (
        latitude.ndim == 1
        and longitude.ndim == 1
        and latitude.size * longitude.size == value_size
    ):
        lat_grid, lon_grid = np.meshgrid(
            latitude,
            longitude,
            indexing="ij",
        )
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
    """Return one request and validate any redundant coordinate arguments."""
    if request is None:
        if lat is None or lon is None:
            raise ValueError(
                "External inputs require source geocentric-GEO lat/lon "
                "or a shared ExternalInputRequest."
            )
        return ExternalInputRequest.from_geocentric_geo(
            lat,
            lon,
            grid_id=grid_id,
        )

    if (lat is None) != (lon is None):
        raise ValueError("lat and lon must be supplied together.")
    if lat is not None:
        supplied = request.source_grid.coordinate_contract.coordinate_identity(
            lat,
            lon,
        )
        if supplied != request.source_grid.coordinate_identity:
            raise ValueError(
                "Explicit coordinates do not match the shared request source grid."
            )
    return request


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
    scalar_lat, scalar_lon = _paired_sample_coordinates(
        lat,
        lon,
        value_size=hall.size,
    )
    scalar_request = ExternalInputRequest.from_geocentric_geo(
        scalar_lat,
        scalar_lon,
        grid_id=str(grid_id),
        sampling_geometry={"type": "sample_points"},
    )

    u_theta = np.asarray(u_theta).reshape(-1)
    u_phi = np.asarray(u_phi).reshape(-1)
    if u_theta.size != u_phi.size:
        raise ValueError("Wind component arrays must have equal sizes.")
    if wind_lat is None:
        wind_lat = lat
    if wind_lon is None:
        wind_lon = lon
    wind_lat, wind_lon = _paired_sample_coordinates(
        wind_lat,
        wind_lon,
        value_size=u_theta.size,
    )
    wind_source_id = (
        str(grid_id)
        if scalar_lat.size == wind_lat.size
        and scalar_request.source_grid.coordinate_contract.coordinate_identity(
            wind_lat,
            wind_lon,
        )
        == scalar_request.source_grid.coordinate_identity
        else f"{grid_id}-wind"
    )
    wind_request = (
        scalar_request
        if wind_source_id == str(grid_id)
        else ExternalInputRequest.from_geocentric_geo(
            wind_lat,
            wind_lon,
            grid_id=wind_source_id,
            sampling_geometry={"type": "sample_points"},
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
            scalar_request.source_grid.grid_id: ProviderDataset(
                spec=CONDUCTANCE_PROVIDER_SPEC,
                source_grid=scalar_request.source_grid,
                request_grid=scalar_request.grid_for(CONDUCTANCE_PROVIDER_SPEC),
                values={"hall": hall, "pedersen": pedersen},
            )
        },
        BOUNDARY_JR_PROVIDER_SPEC.key: {
            scalar_request.source_grid.grid_id: ProviderDataset(
                spec=BOUNDARY_JR_PROVIDER_SPEC,
                source_grid=scalar_request.source_grid,
                request_grid=scalar_request.grid_for(BOUNDARY_JR_PROVIDER_SPEC),
                values={"jr": jr},
            )
        },
        NEUTRAL_WIND_PROVIDER_SPEC.key: {
            wind_request.source_grid.grid_id: ProviderDataset(
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


def _dataset_description(dataset: ProviderDataset) -> str:
    """Return a compact physical-grid description."""
    geometry = dataset.source_grid.sampling_geometry
    provenance = dataset.source_grid.provenance
    origin = provenance.get("originating_model_frame", {})
    details = [
        str(origin.get("horizontal_coordinate_system", "runtime"))
    ]
    if "ncs" in geometry:
        details.append(f"Ncs={geometry['ncs']}")
    if "epoch" in origin:
        details.append(f"epoch={float(origin['epoch']):.6f}")
    details.append(
        dataset.source_grid.coordinate_contract.coordinate_system
    )
    details.append(f"{dataset.source_grid.size} ordered points")
    return f"{dataset.source_grid.grid_id} ({', '.join(details)})"


def _dataset_sort_key(
    item: tuple[str, ProviderDataset],
) -> tuple[Any, ...]:
    """Return a natural diagnostic ordering for cached source grids."""
    source_grid_id, dataset = item
    geometry = dataset.source_grid.sampling_geometry
    origin = dataset.source_grid.provenance.get(
        "originating_model_frame",
        {},
    )
    horizontal = str(origin.get("horizontal_coordinate_system", ""))
    geometry_order = {
        "geographic": 0,
        "centered_dipole_magnetic": 1,
    }.get(horizontal, 2)
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
    entries: Mapping[str, ProviderDataset],
    request: ExternalInputRequest,
    quantity: str,
    *,
    spec: ProviderSpec,
) -> ProviderDataset:
    """Select an exact source/request-grid pair for one provider."""
    if not entries:
        raise ValueError(f"No fallback {quantity} data available.")

    expected_request_grid = request.grid_for(spec)
    compatible = []
    for source_grid_id, dataset in entries.items():
        if dataset.spec != spec:
            continue
        compatible.append((source_grid_id, dataset))
        if (
            dataset.source_grid.coordinate_identity
            == request.source_grid.coordinate_identity
        ):
            if (
                dataset.request_grid.coordinate_identity
                != expected_request_grid.coordinate_identity
            ):
                raise RuntimeError(
                    f"Fallback {quantity} provider-request grid is stale."
                )
            return dataset

    if not compatible:
        raise ValueError(
            f"No fallback {quantity} dataset uses provider specification "
            f"{spec.key!r} ({spec.signature})."
        )

    available = "; ".join(
        _dataset_description(dataset)
        for _, dataset in sorted(compatible, key=_dataset_sort_key)
    )
    raise ValueError(
        f"No fallback {quantity} data matches the requested ordered "
        f"{request.source_grid.size}-point source grid. "
        f"Available compatible grids: {available}"
    )


def get_conductance_inputs(
    date: Any,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    time: np.ndarray | None,
    *,
    request: ExternalInputRequest | None = None,
    kp: float = 5.0,
    starlight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Hardy/EUV conductance on the source PynaMIT grid."""
    request = _coerce_request(
        lat,
        lon,
        request,
        grid_id="runtime-conductance-source",
    )
    source_grid = request.source_grid
    provider_grid = request.grid_for(CONDUCTANCE_PROVIDER_SPEC)
    conductance = _load_optional_module("conductance", "lompe")

    if conductance is not None:
        hall, pedersen = conductance.hardy_EUV(
            provider_grid.lon,
            provider_grid.lat,
            float(kp),
            date,
            starlight=float(starlight),
            dipole=False,
        )
        return (
            _expand_time_series(hall, time),
            _expand_time_series(pedersen, time),
            np.array(source_grid.lat, copy=True),
            np.array(source_grid.lon, copy=True),
        )

    if get_input_source() == "fallback":
        if not np.isclose(float(kp), 5.0) or not np.isclose(float(starlight), 1.0):
            raise ValueError(
                "Bundled conductance fallback data use kp=5 and starlight=1; "
                "custom provider parameters require native inputs."
            )
        collection = _load_fallback()
        dataset = _select_fallback_entry(
            collection.datasets[CONDUCTANCE_PROVIDER_SPEC.key],
            request,
            "conductance",
            spec=collection.providers[CONDUCTANCE_PROVIDER_SPEC.key],
        )
        return (
            _expand_time_series(dataset.values["hall"], time),
            _expand_time_series(dataset.values["pedersen"], time),
            np.array(dataset.source_grid.lat, copy=True),
            np.array(dataset.source_grid.lon, copy=True),
        )

    raise RuntimeError("Native conductance inputs are not available.")


def get_jr_inputs(
    date: Any,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    time: np.ndarray | None,
    *,
    request: ExternalInputRequest | None = None,
    amps_parameters: tuple[float, float, float, float, float] = (
        300.0, 0.0, -4.0, 20.0, 100.0
    ),
    minlat: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AMPS upward radial current on the source PynaMIT grid."""
    request = _coerce_request(
        lat,
        lon,
        request,
        grid_id="runtime-boundary-jr-source",
    )
    source_grid = request.source_grid
    provider_grid = request.grid_for(BOUNDARY_JR_PROVIDER_SPEC)
    pyamps = _load_optional_module("pyamps", "pyamps")

    if pyamps is not None:
        apex = apexpy.Apex(date=date, refh=_HWM_ALTITUDE_KM)
        mlat, mlon = apex.geo2apex(
            provider_grid.lat,
            provider_grid.lon,
            _HWM_ALTITUDE_KM,
        )
        mlt = pyamps.mlon_to_mlt(mlon, date, decimal_year(date))
        coeff_path = os.path.join(
            os.path.dirname(pyamps.__file__),
            "coefficients",
            "SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0104.txt",
        )
        if len(amps_parameters) != 5:
            raise ValueError("amps_parameters must contain five values.")
        amps = pyamps.AMPS(
            *(float(value) for value in amps_parameters),
            minlat=float(minlat),
            coeff_fn=coeff_path,
        )
        jr = amps.get_upward_current(mlat=mlat, mlt=mlt) * 1e-6
        jr[np.abs(mlat) < float(minlat)] = 0
        return (
            _expand_time_series(jr, time),
            np.array(source_grid.lat, copy=True),
            np.array(source_grid.lon, copy=True),
        )

    if get_input_source() == "fallback":
        default_parameters = (300.0, 0.0, -4.0, 20.0, 100.0)
        if (
            tuple(float(value) for value in amps_parameters)
            != default_parameters
            or not np.isclose(float(minlat), 50.0)
        ):
            raise ValueError(
                "Bundled AMPS fallback data use the default AMPS parameters and minlat=50; "
                "custom provider parameters require native inputs."
            )
        collection = _load_fallback()
        dataset = _select_fallback_entry(
            collection.datasets[BOUNDARY_JR_PROVIDER_SPEC.key],
            request,
            "boundary_jr",
            spec=collection.providers[BOUNDARY_JR_PROVIDER_SPEC.key],
        )
        return (
            _expand_time_series(dataset.values["jr"], time),
            np.array(dataset.source_grid.lat, copy=True),
            np.array(dataset.source_grid.lon, copy=True),
        )

    raise RuntimeError("Native FAC/jr inputs are not available.")


def _hwm_iyd(date: Any) -> int:
    """Return HWM's YYDDD integer date code."""
    return int(date.year % 100) * 1000 + int(date.timetuple().tm_yday)


def _hwm_utc_hours(date: Any) -> float:
    """Return UTC hours including seconds and microseconds."""
    return float(
        date.hour
        + date.minute / 60.0
        + date.second / 3600.0
        + date.microsecond / 3.6e9
    )


def _library_horizontal_wind_to_spherical(
    request: ExternalInputRequest,
    zonal_east: np.ndarray,
    meridional_north: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the spherical east/north identity used by all adapters."""
    source_grid = request.source_grid
    provider_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    zonal_east, meridional_north = np.broadcast_arrays(
        np.asarray(zonal_east, dtype=float),
        np.asarray(meridional_north, dtype=float),
    )
    if zonal_east.size != source_grid.size:
        raise ValueError("HWM wind values must match the request grid size.")
    if not np.array_equal(provider_grid.lat, source_grid.lat):
        raise RuntimeError("Library request latitude must equal source latitude.")
    if not np.array_equal(provider_grid.lon, source_grid.lon):
        raise RuntimeError("Library request longitude must equal source longitude.")
    u_theta, u_phi = library_horizontal_to_spherical(
        zonal_east.reshape(-1),
        meridional_north.reshape(-1),
    )
    return u_theta, u_phi


def get_wind_inputs(
    date: Any,
    use_wind: bool,
    time: np.ndarray | None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    *,
    request: ExternalInputRequest | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
] | None:
    """Return HWM wind on the same source grid as other empirical inputs."""
    if not use_wind:
        return None

    request = _coerce_request(
        lat,
        lon,
        request,
        grid_id="runtime-neutral-wind-source",
    )
    source_grid = request.source_grid
    provider_grid = request.grid_for(NEUTRAL_WIND_PROVIDER_SPEC)
    pyhwm2014 = _load_optional_module("pyhwm2014", "pyhwm2014")

    if pyhwm2014 is not None:
        evaluator = getattr(pyhwm2014, "hwm14_vectorized", None)
        if not callable(evaluator):
            raise RuntimeError(
                "pyhwm2014.hwm14_vectorized is required; update the "
                "pyHWM14 main-branch dependency."
            )
        zonal_east, meridional_north = evaluator(
            alt_km=np.full(provider_grid.size, _HWM_ALTITUDE_KM),
            glat_deg=provider_grid.lat,
            glon_deg=provider_grid.lon,
            utc_hours=np.full(provider_grid.size, _hwm_utc_hours(date)),
            iyd=_hwm_iyd(date),
            ap=list(_HWM_AP),
        )
        u_theta, u_phi = _library_horizontal_wind_to_spherical(
            request,
            zonal_east,
            meridional_north,
        )
        return (
            _expand_time_series(u_theta, time),
            _expand_time_series(u_phi, time),
            np.array(source_grid.lat, copy=True),
            np.array(source_grid.lon, copy=True),
            None,
        )

    if get_input_source() != "fallback":
        raise RuntimeError("Native neutral-wind inputs are not available.")

    collection = _load_fallback()
    dataset = _select_fallback_entry(
        collection.datasets[NEUTRAL_WIND_PROVIDER_SPEC.key],
        request,
        "neutral_wind",
        spec=collection.providers[NEUTRAL_WIND_PROVIDER_SPEC.key],
    )
    return (
        _expand_time_series(dataset.values["u_theta"], time),
        _expand_time_series(dataset.values["u_phi"], time),
        np.array(dataset.source_grid.lat, copy=True),
        np.array(dataset.source_grid.lon, copy=True),
        None,
    )
