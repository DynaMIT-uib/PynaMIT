"""Ground magnetometer station data access and preparation."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from kompe.cache import BoundedCache

from pynamit.results.time_series import (
    compute_centered_difference_series_at_times,
    resample_series_to_times,
)

_STATION_FILE_CACHE = BoundedCache(512)


def station_component_columns(station_code):
    """Return standardized XYZ column names for an IAGA station code."""
    station_code = str(station_code).upper()
    return [f"{station_code}X", f"{station_code}Y", f"{station_code}Z"]


def shift_station_datetime_index(time_index, data_time_offset_seconds=0.0):
    """Shift station timestamps by the configured data offset."""
    data_index = pd.DatetimeIndex(pd.to_datetime(time_index))
    offset_seconds = float(data_time_offset_seconds)
    if abs(offset_seconds) <= 1e-12:
        return data_index
    return data_index + pd.to_timedelta(offset_seconds, unit="s")


def station_source_time_window(start_time, end_time, data_time_offset_seconds=0.0):
    """Return the source-file time window for shifted display times."""
    data_offset = pd.to_timedelta(float(data_time_offset_seconds), unit="s")
    return pd.Timestamp(start_time) - data_offset, pd.Timestamp(end_time) - data_offset


def normalize_station_catalog(stations_df):
    """Normalize an IAGA station catalog."""
    normalized = stations_df.copy()
    normalized["IAGA"] = normalized["IAGA"].astype(str).str.upper()
    normalized["GEOLAT"] = pd.to_numeric(normalized["GEOLAT"], errors="coerce")
    lon = pd.to_numeric(normalized["GEOLON"], errors="coerce")
    normalized["GEOLON"] = ((lon + 180.0) % 360.0) - 180.0
    return normalized.reset_index(drop=True)


def load_station_catalog(simulation_directory, station_data_directory=None):
    """Load a normalized station catalog and return its source path."""
    simulation_directory = Path(simulation_directory).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []
    if station_data_directory:
        candidates.append(Path(station_data_directory).expanduser() / "stations_full_list.csv")
    candidates.extend(
        [
            simulation_directory / "mag_data" / "stations_full_list.csv",
            simulation_directory / "data" / "mag_data" / "stations_full_list.csv",
            Path("mag_data/stations_full_list.csv"),
            Path("notebooks/mag_data/stations_full_list.csv"),
            repo_root / "notebooks" / "mag_data" / "stations_full_list.csv",
        ]
    )
    for candidate in candidates:
        try:
            return normalize_station_catalog(pd.read_csv(candidate)), candidate
        except FileNotFoundError:
            continue
    raise ValueError(
        "Could not find stations_full_list.csv. Set station_data_directory in "
        "pynamit_plot_defaults.json or place station data in mag_data/."
    )


def _log(logger, message):
    if logger is not None:
        logger(message)


def _iaga2002_download_url(station_code, sim_start_time):
    date_str_url = pd.Timestamp(sim_start_time).strftime("%Y-%m-%d")
    return (
        "https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=Iaga2002"
        f"&testObsys=0&observatoryIagaCode={str(station_code).upper()}&samplesPerDay=second"
        f"&publicationState=Best%20available&dataStartDate={date_str_url}"
        "&dataDuration=1&orientation=native"
    )


def load_iaga2002_magnetometer_data(filepath, station_code, *, logger=None):
    """
    Load IAGA2002 data and convert supported components to XYZ.

    Missing IAGA flags are converted to NaN.
    The result uses a datetime index and standardized XYZ columns.
    """
    filepath = Path(filepath)
    station_prefix = str(station_code).upper()
    try:
        with filepath.open("r") as file:
            for line_number, line in enumerate(file):
                if line.startswith("DATE"):
                    header_line_num = line_number
                    header_content = line.strip().split()
                    break
            else:
                _log(logger, f"Format Error: Data header not found in '{filepath}'.")
                return None

        available_components = [
            component.replace(station_prefix, "")
            for component in header_content
            if component.startswith(station_prefix)
        ]
        if not available_components:
            _log(
                logger,
                "Format Error: No data columns found for station "
                f"{station_prefix} in '{filepath}'.",
            )
            return None

        use_cols = ["DATE", "TIME"] + [
            f"{station_prefix}{component}" for component in available_components
        ]
        data = pd.read_csv(
            filepath,
            skiprows=header_line_num,
            header=0,
            sep=r"\s+",
            usecols=lambda column: column.strip() in use_cols,
            na_values=[99999.0, 88888.0, 99999.9, 88888.8],
        )

        if not all(component in available_components for component in ["X", "Y", "Z"]):
            if "D" not in available_components or "H" not in available_components:
                _log(
                    logger,
                    f"Format Error: Unsupported magnetic components {available_components} "
                    f"in '{filepath}'. Need XYZ or DHI/DHZ.",
                )
                return None

            d_radians = np.radians(data[f"{station_prefix}D"] / 600.0)
            h_nt = data[f"{station_prefix}H"]
            data[f"{station_prefix}X"] = h_nt * np.cos(d_radians)
            data[f"{station_prefix}Y"] = h_nt * np.sin(d_radians)

            if "Z" not in available_components:
                if "I" not in available_components:
                    _log(
                        logger,
                        "Format Error: Cannot calculate Z component. "
                        f"Need 'Z' or 'I' column in '{filepath}'.",
                    )
                    return None
                i_radians = np.radians(data[f"{station_prefix}I"] / 600.0)
                data[f"{station_prefix}Z"] = h_nt * np.tan(i_radians)

        final_cols = station_component_columns(station_prefix)
        if not all(column in data.columns for column in final_cols):
            _log(
                logger,
                f"Format Error: Could not produce all required XYZ columns in '{filepath}'.",
            )
            return None

        data["datetime"] = pd.to_datetime(data["DATE"] + " " + data["TIME"])
        data = data.set_index("datetime").drop(columns=["DATE", "TIME"])
        return data[final_cols]

    except (KeyError, OSError, TypeError, UnicodeError, ValueError) as exc:
        _log(logger, f"Error loading and processing magnetometer data from {filepath}: {exc}")
        return None


def load_local_iaga2002_station_data(
    data_directory, station_code, display_time, *, data_time_offset_seconds=0.0
):
    """Load a local IAGA2002 day in geographic components."""
    source_time = pd.Timestamp(display_time) - pd.to_timedelta(
        float(data_time_offset_seconds), unit="s"
    )
    path = (
        Path(data_directory)
        / f"{str(station_code).lower()}{source_time.strftime('%Y%m%d')}vsec.sec"
    )
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return None
    cache_key = (
        str(path.resolve()),
        file_stat.st_mtime_ns,
        file_stat.st_size,
        str(station_code).upper(),
    )
    measured = _STATION_FILE_CACHE.get_or_create(
        cache_key, lambda: load_iaga2002_magnetometer_data(path, station_code, logger=None)
    )
    if measured is None:
        return None
    return iaga_xyz_to_geographic_components(
        measured, station_code, data_time_offset_seconds=data_time_offset_seconds
    )


def iaga_xyz_to_geographic_components(measured, station_code, *, data_time_offset_seconds=0.0):
    """Convert IAGA XYZ columns to geographic north, east, and down."""
    station_code = str(station_code).upper()
    measured_index = shift_station_datetime_index(
        measured.index, data_time_offset_seconds=data_time_offset_seconds
    )
    return pd.DataFrame(
        {
            "North": measured[f"{station_code}X"].to_numpy(dtype=float),
            "East": measured[f"{station_code}Y"].to_numpy(dtype=float),
            "Down": measured[f"{station_code}Z"].to_numpy(dtype=float),
        },
        index=measured_index,
    )


def download_and_load_iaga2002_station_data(
    station_code, sim_start_time, data_dir, *, logger=None
):
    """Load local IAGA2002 data, downloading when missing."""
    station_code = str(station_code).upper()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    date_str_file = pd.Timestamp(sim_start_time).strftime("%Y%m%d")
    filename = data_dir / f"{station_code.lower()}{date_str_file}vsec.sec"

    if not filename.exists():
        _log(logger, f"File '{filename.name}' not found. Downloading from BGS...")
        try:
            with urllib.request.urlopen(
                _iaga2002_download_url(station_code, sim_start_time)
            ) as response:
                data = response.read()
        except urllib.error.URLError:
            _log(logger, f"Failed to download data for {station_code}.")
            return None

        if data.strip().startswith(b"<!DOCTYPE") or b"Error" in data[:200]:
            _log(logger, f"Error: Could not retrieve data for {station_code}.")
            return None

        filename.write_bytes(data)
        _log(logger, "Download complete.")

    return load_iaga2002_magnetometer_data(filename, station_code, logger=logger)


def station_window_has_nonzero_measurements(
    mag_df_full, station_code, start_time, end_time, *, data_time_offset_seconds=0.0
):
    """Return True when all XYZ station components are nonzero."""
    if mag_df_full is None:
        return False
    mag_cols = station_component_columns(station_code)
    if not all(column in mag_df_full.columns for column in mag_cols):
        return False
    source_start_time, source_end_time = station_source_time_window(
        start_time, end_time, data_time_offset_seconds
    )
    window_df = mag_df_full.loc[source_start_time:source_end_time, mag_cols]
    if window_df.empty:
        return False
    for column in mag_cols:
        values = window_df[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0 or not np.any(np.abs(values) > np.finfo(float).tiny):
            return False
    return True


def station_has_complete_nonzero_components_at_times(
    mag_df_full,
    station_code,
    target_times,
    *,
    plot_dbdt=False,
    half_window_points=None,
    cadence_seconds=None,
    data_time_offset_seconds=0.0,
):
    """Return True when all XYZ components are finite and nonzero."""
    if mag_df_full is None:
        return False
    mag_cols = station_component_columns(station_code)
    if not all(column in mag_df_full.columns for column in mag_cols):
        return False
    measured_time_index = shift_station_datetime_index(
        mag_df_full.index, data_time_offset_seconds=data_time_offset_seconds
    )
    target_index = pd.DatetimeIndex(pd.to_datetime(target_times))
    if len(target_index) == 0:
        return False
    for column in mag_cols:
        if plot_dbdt:
            values = compute_centered_difference_series_at_times(
                measured_time_index,
                mag_df_full[column].to_numpy(dtype=float),
                target_index,
                half_window_points=half_window_points,
                cadence_seconds=cadence_seconds,
            )
        else:
            values = resample_series_to_times(
                measured_time_index, mag_df_full[column].to_numpy(dtype=float), target_index
            )
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != len(target_index) or not np.all(np.isfinite(values)):
            return False
        if not np.any(np.abs(values) > np.finfo(float).tiny):
            return False
    return True


__all__ = [
    "download_and_load_iaga2002_station_data",
    "iaga_xyz_to_geographic_components",
    "load_iaga2002_magnetometer_data",
    "load_local_iaga2002_station_data",
    "load_station_catalog",
    "normalize_station_catalog",
    "shift_station_datetime_index",
    "station_component_columns",
    "station_has_complete_nonzero_components_at_times",
    "station_source_time_window",
    "station_window_has_nonzero_measurements",
]
