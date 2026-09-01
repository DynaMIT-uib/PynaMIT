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

from pynamit.coordinates import datetime_to_utc_hours, decimal_year
from pynamit.external_inputs.coordinates import (
    LIBRARY_GEOGRAPHIC_110KM,
    ExternalInputCoordinates,
    SampleGrid,
)
from pynamit.external_inputs.fallback_data import (
    FALLBACK_SCHEMA_VERSION,
    FallbackCollection,
    ProviderSnapshot,
)
from pynamit.external_inputs.provider_definitions import (
    BOUNDARY_JR_PROVIDER_SPEC,
    CONDUCTANCE_PROVIDER_SPEC,
    NEUTRAL_WIND_PROVIDER_SPEC,
    PROVIDER_SPECS,
    InputProviderSpec,
)
from pynamit.geodesy import library_horizontal_to_spherical

FALLBACK_RESOURCE = resources.files("pynamit.data") / "fallback_inputs.json"
_INPUT_SOURCE = os.environ.get("PYNAMIT_INPUT_SOURCE", "native").strip().lower()
if _INPUT_SOURCE == "auto":
    _INPUT_SOURCE = "native"
if _INPUT_SOURCE not in {"fallback", "native"}:
    raise ValueError("PYNAMIT_INPUT_SOURCE must be 'fallback' or 'native'.")

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


def require_native_inputs() -> None:
    """Import native providers, raising when unusable."""
    lompe = import_module("lompe")
    _ = lompe.conductance
    import_module("pyamps")
    hwm = import_module("pyhwm2014")
    if not callable(getattr(hwm, "hwm14_vectorized", None)):
        raise ImportError("pyhwm2014 does not provide hwm14_vectorized.")


def native_inputs_available() -> bool:
    """Return whether all native empirical providers are importable."""
    try:
        require_native_inputs()
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


def _coerce_coordinates(
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    coordinates: ExternalInputCoordinates | None,
    *,
    grid_id: str,
) -> ExternalInputCoordinates:
    """Return input coordinates after validating redundant arrays."""
    if coordinates is None:
        if lat is None or lon is None:
            raise ValueError(
                "External inputs require geocentric-GEO lat/lon "
                "or a shared ExternalInputCoordinates."
            )
        return ExternalInputCoordinates.from_geocentric_geo(lat, lon, grid_id=grid_id)

    if (lat is None) != (lon is None):
        raise ValueError("lat and lon must be supplied together.")
    if lat is not None:
        supplied = coordinates.geographic_grid.coordinate_convention.coordinate_identity(lat, lon)
        if supplied != coordinates.geographic_grid.coordinate_identity:
            raise ValueError("Explicit lat/lon do not match the shared geographic grid.")
    return coordinates


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
    scalar_coordinates = ExternalInputCoordinates.from_geocentric_geo(
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
    wind_grid_id = (
        str(grid_id)
        if scalar_lat.size == wind_lat.size
        and scalar_coordinates.geographic_grid.coordinate_convention.coordinate_identity(
            wind_lat, wind_lon
        )
        == scalar_coordinates.geographic_grid.coordinate_identity
        else f"{grid_id}-wind"
    )
    wind_coordinates = (
        scalar_coordinates
        if wind_grid_id == str(grid_id)
        else ExternalInputCoordinates.from_geocentric_geo(
            wind_lat, wind_lon, grid_id=wind_grid_id, sampling_geometry={"type": "sample_points"}
        )
    )

    grids: dict[str, SampleGrid] = {}
    for coordinate_set in (scalar_coordinates, wind_coordinates):
        geographic_grid = coordinate_set.geographic_grid
        provider_grid = coordinate_set.sample_grid(LIBRARY_GEOGRAPHIC_110KM)
        grids[geographic_grid.grid_id] = geographic_grid
        grids[provider_grid.grid_id] = provider_grid

    datasets = {
        CONDUCTANCE_PROVIDER_SPEC.key: {
            scalar_coordinates.geographic_grid.grid_id: ProviderSnapshot(
                spec=CONDUCTANCE_PROVIDER_SPEC,
                geographic_grid=scalar_coordinates.geographic_grid,
                request_grid=scalar_coordinates.sample_grid(
                    CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
                ),
                values={"hall": hall, "pedersen": pedersen},
            )
        },
        BOUNDARY_JR_PROVIDER_SPEC.key: {
            scalar_coordinates.geographic_grid.grid_id: ProviderSnapshot(
                spec=BOUNDARY_JR_PROVIDER_SPEC,
                geographic_grid=scalar_coordinates.geographic_grid,
                request_grid=scalar_coordinates.sample_grid(
                    BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_convention
                ),
                values={"jr": jr},
            )
        },
        NEUTRAL_WIND_PROVIDER_SPEC.key: {
            wind_coordinates.geographic_grid.grid_id: ProviderSnapshot(
                spec=NEUTRAL_WIND_PROVIDER_SPEC,
                geographic_grid=wind_coordinates.geographic_grid,
                request_grid=wind_coordinates.sample_grid(
                    NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention
                ),
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


def _dataset_description(dataset: ProviderSnapshot) -> str:
    """Return a compact physical-grid description."""
    geometry = dataset.geographic_grid.sampling_geometry
    provenance = dataset.geographic_grid.provenance
    origin = provenance.get("originating_model_frame", {})
    details = [str(origin.get("horizontal_coordinate_system", "runtime"))]
    if "ncs" in geometry:
        details.append(f"Ncs={geometry['ncs']}")
    if "epoch" in origin:
        details.append(f"epoch={float(origin['epoch']):.6f}")
    details.append(dataset.geographic_grid.coordinate_convention.coordinate_system)
    details.append(f"{dataset.geographic_grid.size} ordered points")
    return f"{dataset.geographic_grid.grid_id} ({', '.join(details)})"


def _dataset_sort_key(item: tuple[str, ProviderSnapshot]) -> tuple[Any, ...]:
    """Return the diagnostic ordering for cached geographic grids."""
    geographic_grid_id, dataset = item
    geometry = dataset.geographic_grid.sampling_geometry
    origin = dataset.geographic_grid.provenance.get("originating_model_frame", {})
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
    return geometry_order, ncs, epoch, str(geographic_grid_id)


def _select_fallback_entry(
    entries: Mapping[str, ProviderSnapshot],
    coordinates: ExternalInputCoordinates,
    quantity: str,
    *,
    spec: InputProviderSpec,
) -> ProviderSnapshot:
    """Select an exact geographic/request-grid pair for one provider."""
    if not entries:
        raise ValueError(f"No fallback {quantity} data available.")

    expected_request_grid = coordinates.sample_grid(spec.request_coordinate_convention)
    compatible = []
    for geographic_grid_id, dataset in entries.items():
        if dataset.spec != spec:
            continue
        compatible.append((geographic_grid_id, dataset))
        if (
            dataset.geographic_grid.coordinate_identity
            == coordinates.geographic_grid.coordinate_identity
        ):
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
        f"{coordinates.geographic_grid.size}-point geographic grid. "
        f"Available compatible grids: {available}"
    )


def _validated_centered_dipole(coordinates: ExternalInputCoordinates) -> dipole.Dipole:
    """Return the dipole after checking the paired geographic view."""
    model_epoch = coordinates.model_epoch
    if model_epoch is None:
        raise ValueError("A centered-dipole external-input view requires a model epoch.")

    model_grid = coordinates.model_grid
    geographic_grid = coordinates.geographic_grid
    centered_dipole = dipole.Dipole(model_epoch)
    geographic_lat, geographic_lon = centered_dipole.mag2geo(model_grid.lat, model_grid.lon)
    longitude_error = (np.asarray(geographic_lon) - geographic_grid.lon + 180.0) % 360.0 - 180.0
    if not np.allclose(
        geographic_lat, geographic_grid.lat, rtol=0.0, atol=1e-8
    ) or not np.allclose(longitude_error, 0.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            "Centered-dipole model coordinates, model_epoch, and physical GEO samples disagree."
        )
    return centered_dipole


def _hardy_euv_from_coordinate_views(
    conductance,
    coordinates: ExternalInputCoordinates,
    provider_date: dt.datetime,
    centered_dipole: dipole.Dipole | None,
    *,
    kp: int,
    starlight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate Hardy magnetically and EUV in physical GEO."""
    model_grid = coordinates.model_grid
    geographic_grid = coordinates.geographic_grid
    if centered_dipole is None:
        provider_grid = coordinates.sample_grid(
            CONDUCTANCE_PROVIDER_SPEC.request_coordinate_convention
        )
        apex = apexpy.Apex(date=provider_date, refh=_IONOSPHERE_ALTITUDE_KM)
        auroral_lat, auroral_lon = apex.geo2apex(
            provider_grid.lat, provider_grid.lon, _IONOSPHERE_ALTITUDE_KM
        )
        centered_dipole = dipole.Dipole(decimal_year(provider_date))
    else:
        auroral_lat, auroral_lon = model_grid.lat, model_grid.lon

    mlt = centered_dipole.mlon2mlt(auroral_lon, provider_date)
    solar_zenith_angle = conductance.sunlight.sza(
        geographic_grid.lat, geographic_grid.lon, provider_date
    )
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
    coordinates: ExternalInputCoordinates | None = None,
    kp: int,
    starlight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Pedersen/Hall conductance on the geographic PynaMIT grid.

    ``kp`` is the Hardy activity index and ``starlight`` is the
    nightside background conductance in siemens.

    Returns
    -------
    pedersen, hall, lat, lon : ndarray
        Conductance in siemens followed by geographic-grid coordinates.
        Lompe's native Hardy/EUV routine returns Hall first; that order
        is converted explicitly at this adapter boundary.
    """
    coordinates = _coerce_coordinates(
        lat, lon, coordinates, grid_id="runtime-conductance-geographic"
    )
    geographic_grid = coordinates.geographic_grid
    centered_dipole_convention = CONDUCTANCE_PROVIDER_SPEC.request_coordinate_views["model"]
    centered_dipole = (
        _validated_centered_dipole(coordinates)
        if coordinates.model_grid.coordinate_convention == centered_dipole_convention
        else None
    )
    conductance = _load_optional_module("conductance", "lompe")

    if conductance is not None:
        provider_date = _provider_utc_datetime(date)
        hall, pedersen = _hardy_euv_from_coordinate_views(
            conductance,
            coordinates,
            provider_date,
            centered_dipole,
            kp=int(kp),
            starlight=float(starlight),
        )
        hall = _provider_values(
            hall, provider="Lompe Hardy/EUV", field="hall", expected_size=geographic_grid.size
        )
        pedersen = _provider_values(
            pedersen,
            provider="Lompe Hardy/EUV",
            field="pedersen",
            expected_size=geographic_grid.size,
        )
        return (
            pedersen,
            hall,
            np.array(geographic_grid.lat, copy=True),
            np.array(geographic_grid.lon, copy=True),
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
            coordinates,
            "conductance",
            spec=collection.providers[CONDUCTANCE_PROVIDER_SPEC.key],
        )
        return (
            np.array(dataset.values["pedersen"], copy=True),
            np.array(dataset.values["hall"], copy=True),
            np.array(dataset.geographic_grid.lat, copy=True),
            np.array(dataset.geographic_grid.lon, copy=True),
        )

    raise RuntimeError("Native conductance inputs are not available.")


def get_boundary_jr_inputs(
    date: Any,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    *,
    coordinates: ExternalInputCoordinates | None = None,
    v: float,
    By: float,
    Bz: float,
    tilt: float,
    f107: float,
    minlat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AMPS upward radial current on the geographic PynaMIT grid.

    The AMPS drivers are solar-wind speed ``v`` in km/s, IMF ``By`` and
    ``Bz`` in nT, dipole ``tilt`` in degrees, F10.7 flux ``f107`` in
    sfu, and the model cutoff ``minlat`` in degrees.
    """
    coordinates = _coerce_coordinates(
        lat, lon, coordinates, grid_id="runtime-boundary-jr-geographic"
    )
    geographic_grid = coordinates.geographic_grid
    centered_dipole_convention = BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_views["model"]
    centered_dipole = (
        _validated_centered_dipole(coordinates)
        if coordinates.model_grid.coordinate_convention == centered_dipole_convention
        else None
    )
    pyamps = _load_optional_module("pyamps", "pyamps")

    if pyamps is not None:
        provider_date = _provider_utc_datetime(date)
        if centered_dipole is None:
            provider_grid = coordinates.sample_grid(
                BOUNDARY_JR_PROVIDER_SPEC.request_coordinate_convention
            )
            apex = apexpy.Apex(date=provider_date, refh=_IONOSPHERE_ALTITUDE_KM)
            mlat, mlon = apex.geo2apex(
                provider_grid.lat, provider_grid.lon, _IONOSPHERE_ALTITUDE_KM
            )
            mlt = pyamps.mlon_to_mlt(mlon, provider_date, decimal_year(provider_date))
        else:
            mlat, mlon = coordinates.model_grid.lat, coordinates.model_grid.lon
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
                expected_size=geographic_grid.size,
            )
            * 1e-6
        )
        jr[np.abs(mlat) < float(minlat)] = 0
        return (
            jr,
            np.array(geographic_grid.lat, copy=True),
            np.array(geographic_grid.lon, copy=True),
        )

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
            coordinates,
            "boundary_jr",
            spec=collection.providers[BOUNDARY_JR_PROVIDER_SPEC.key],
        )
        return (
            np.array(dataset.values["jr"], copy=True),
            np.array(dataset.geographic_grid.lat, copy=True),
            np.array(dataset.geographic_grid.lon, copy=True),
        )

    raise RuntimeError("Native FAC/jr inputs are not available.")


def _hwm_iyd(date: Any) -> int:
    """Return HWM's YYDDD integer date code."""
    return int(date.year % 100) * 1000 + int(date.timetuple().tm_yday)


def _library_horizontal_wind_to_spherical(
    coordinates: ExternalInputCoordinates, zonal_east: np.ndarray, meridional_north: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the spherical east/north identity used by all adapters."""
    geographic_grid = coordinates.geographic_grid
    provider_grid = coordinates.sample_grid(
        NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention
    )
    zonal_east, meridional_north = np.broadcast_arrays(
        np.asarray(zonal_east, dtype=float), np.asarray(meridional_north, dtype=float)
    )
    if zonal_east.size != geographic_grid.size:
        raise ValueError("HWM wind values must match the request grid size.")
    if not np.array_equal(provider_grid.lat, geographic_grid.lat):
        raise RuntimeError("Library coordinates latitude must equal geographic latitude.")
    if not np.array_equal(provider_grid.lon, geographic_grid.lon):
        raise RuntimeError("Library coordinates longitude must equal geographic longitude.")
    u_theta, u_phi = library_horizontal_to_spherical(
        zonal_east.reshape(-1), meridional_north.reshape(-1)
    )
    return u_theta, u_phi


def get_wind_inputs(
    date: Any,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    *,
    coordinates: ExternalInputCoordinates | None = None,
    ap: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return HWM wind on the shared empirical-input geographic grid.

    ``ap`` is the two-value geomagnetic-activity input expected by the
    vectorized pyHWM2014 interface.
    """
    coordinates = _coerce_coordinates(
        lat, lon, coordinates, grid_id="runtime-neutral-wind-geographic"
    )
    geographic_grid = coordinates.geographic_grid
    provider_grid = coordinates.sample_grid(
        NEUTRAL_WIND_PROVIDER_SPEC.request_coordinate_convention
    )
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
            utc_hours=np.full(provider_grid.size, datetime_to_utc_hours(provider_date)),
            iyd=_hwm_iyd(provider_date),
            ap=list(ap),
        )
        u_theta, u_phi = _library_horizontal_wind_to_spherical(
            coordinates, zonal_east, meridional_north
        )
        return (
            u_theta,
            u_phi,
            np.array(geographic_grid.lat, copy=True),
            np.array(geographic_grid.lon, copy=True),
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
        coordinates,
        "neutral_wind",
        spec=collection.providers[NEUTRAL_WIND_PROVIDER_SPEC.key],
    )
    return (
        np.array(dataset.values["u_theta"], copy=True),
        np.array(dataset.values["u_phi"], copy=True),
        np.array(dataset.geographic_grid.lat, copy=True),
        np.array(dataset.geographic_grid.lon, copy=True),
        None,
    )
