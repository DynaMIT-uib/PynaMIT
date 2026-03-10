"""Internal ground-station postprocess helpers used by notebook-style views."""

from __future__ import annotations

import datetime
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


LogCallback = Optional[Callable[[str], None]]
_COMPONENTS = ("North", "East", "Down")
TimeWindow = tuple[datetime.time, datetime.time]
DEFAULT_BASELINE_WINDOW: TimeWindow = (
    datetime.time(18, 20),
    datetime.time(18, 30),
)
DEFAULT_ANALYSIS_WINDOW: TimeWindow = (
    datetime.time(18, 30),
    datetime.time(18, 35),
)
_IAGA_MAGNETIC_COMPONENTS = frozenset({"X", "Y", "Z", "D", "H", "I"})


def normalize_station_metadata(stations: pd.DataFrame) -> pd.DataFrame:
    """Return station metadata with canonical casing and longitude range.

    The notebook station CSV mixes longitudes in ``[-180, 180]`` and
    ``[0, 360)``. Normalizing once keeps downstream geometry and labeling
    logic simpler and avoids implicit wrap handling in multiple places.
    """
    normalized = stations.copy()
    if "IAGA" in normalized.columns:
        normalized["IAGA"] = normalized["IAGA"].astype(str).str.upper()
    if "GEOLAT" in normalized.columns:
        normalized["GEOLAT"] = pd.to_numeric(normalized["GEOLAT"], errors="coerce")
    if "GEOLON" in normalized.columns:
        lon = pd.to_numeric(normalized["GEOLON"], errors="coerce")
        normalized["GEOLON"] = ((lon + 180.0) % 360.0) - 180.0
    return normalized


@dataclass(frozen=True)
class ComparisonMetrics:
    correlations: dict[str, float]
    rmses: dict[str, float]


@dataclass(frozen=True)
class StationMetrics:
    inductive: ComparisonMetrics
    magnetostatic: ComparisonMetrics
    baseline_offsets: dict[str, float]


class MagnetometerDataArchive:
    """Load and optionally download IAGA2002 station data for one simulation date."""

    def __init__(self, *, data_dir: Path, reference_time: datetime.datetime) -> None:
        self.data_dir = Path(data_dir)
        self.reference_time = reference_time
        self._data_cache: dict[str, Optional[pd.DataFrame]] = {}
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def local_path(self, station_code: str) -> Path:
        station_code = str(station_code).lower()
        date_str = self.reference_time.strftime("%Y%m%d")
        return self.data_dir / f"{station_code}{date_str}vsec.sec"

    def load_local_station_data(
        self,
        station_code: str,
        *,
        silent: bool = True,
        log: LogCallback = None,
    ) -> Optional[pd.DataFrame]:
        station_code = str(station_code).upper()
        path = self.local_path(station_code)
        if not path.exists():
            return None
        if station_code in self._data_cache:
            return self._data_cache[station_code]
        data = load_iaga2002_station_data(path, station_code, silent=silent, log=log)
        self._data_cache[station_code] = data
        return data

    def get_station_data(
        self,
        station_code: str,
        *,
        download_if_missing: bool = False,
        silent: bool = True,
        log: LogCallback = None,
    ) -> Optional[pd.DataFrame]:
        local = self.load_local_station_data(station_code, silent=silent, log=log)
        if local is not None or not download_if_missing:
            return local
        return self.download_and_load_station_data(station_code, silent=silent, log=log)

    def download_and_load_station_data(
        self,
        station_code: str,
        *,
        silent: bool = False,
        log: LogCallback = None,
    ) -> Optional[pd.DataFrame]:
        station_code = str(station_code).upper()
        path = self.local_path(station_code)
        if path.exists():
            data = load_iaga2002_station_data(path, station_code, silent=silent, log=log)
            self._data_cache[station_code] = data
            return data

        date_str_url = self.reference_time.strftime("%Y-%m-%d")
        url = (
            "https://imag-data.bgs.ac.uk/GIN_V1/GINServices?Request=GetData&format=Iaga2002"
            f"&testObsys=0&observatoryIagaCode={station_code}&samplesPerDay=second"
            f"&publicationState=Best%20available&dataStartDate={date_str_url}&dataDuration=1"
            "&orientation=native"
        )
        if not silent and log is not None:
            log(f"File '{path.name}' not found. Downloading from BGS...")
        try:
            with urllib.request.urlopen(url) as response, path.open("wb") as out_file:
                data = response.read()
                if data.strip().startswith(b"<!DOCTYPE") or b"Error" in data[:200]:
                    if not silent and log is not None:
                        log(f"Error: Could not retrieve data for {station_code}.")
                    if path.exists():
                        path.unlink()
                    return None
                out_file.write(data)
        except urllib.error.URLError:
            if not silent and log is not None:
                log(f"Failed to download data for {station_code}.")
            return None

        if not silent and log is not None:
            log("Download complete.")
        loaded = load_iaga2002_station_data(path, station_code, silent=silent, log=log)
        self._data_cache[station_code] = loaded
        return loaded

    def station_status(
        self,
        station_code: str,
        *,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> str:
        station_code = str(station_code).upper()
        path = self.local_path(station_code)
        if not path.exists():
            return "Download"
        data = self.load_local_station_data(station_code, silent=True)
        if data is None:
            return "Error"
        cols = [f"{station_code}X", f"{station_code}Y", f"{station_code}Z"]
        available_cols = [col for col in cols if col in data.columns]
        if not available_cols:
            return "Error"
        window = data.loc[start_time:end_time, available_cols]
        if window.empty or window.dropna(how="all").empty:
            return "No Data"
        return "Data"


@dataclass(frozen=True)
class IAGA2002Header:
    """Minimal parsed IAGA2002 header metadata used for validation."""

    header_line_num: int
    header_content: tuple[str, ...]
    reported_elements: Optional[str]
    sensor_orientation: Optional[str]


def _parse_iaga2002_header(filepath: Path) -> Optional[IAGA2002Header]:
    """Return parsed header metadata up to the data-column header line."""
    header_line_num = None
    header_content: tuple[str, ...] | None = None
    reported_elements: Optional[str] = None
    sensor_orientation: Optional[str] = None

    with filepath.open("r") as handle:
        for idx, line in enumerate(handle):
            stripped = line.lstrip()
            if stripped.startswith("DATE"):
                header_line_num = idx
                header_content = tuple(stripped.strip().split())
                break
            if stripped.startswith("#"):
                continue

            body = stripped.split("|", 1)[0].rstrip()
            if body.startswith("Reported"):
                value = body.removeprefix("Reported").strip().upper()
                reported_elements = value or None
            elif body.startswith("Sensor Orientation"):
                value = body.removeprefix("Sensor Orientation").strip().upper()
                sensor_orientation = value or None

    if header_line_num is None or header_content is None:
        return None
    return IAGA2002Header(
        header_line_num=header_line_num,
        header_content=header_content,
        reported_elements=reported_elements,
        sensor_orientation=sensor_orientation,
    )


def _extract_reported_component_set(elements: Optional[str]) -> Optional[set[str]]:
    """Return magnetic component letters from an IAGA reported-element string."""
    if not elements:
        return None
    letters = {char for char in str(elements).upper() if char in _IAGA_MAGNETIC_COMPONENTS}
    return letters or None


def _validate_iaga_component_metadata(
    *,
    station_prefix: str,
    filepath: Path,
    actual_components: list[str],
    header: IAGA2002Header,
) -> Optional[str]:
    """Return an error string if header metadata conflicts with data columns."""
    reported_components = _extract_reported_component_set(header.reported_elements)
    actual_component_set = {component for component in actual_components if len(component) == 1}
    actual_component_set = {
        component for component in actual_component_set if component in _IAGA_MAGNETIC_COMPONENTS
    }

    if reported_components is not None and actual_component_set != reported_components:
        return (
            "Format Error: Reported elements "
            f"{sorted(reported_components)} do not match data columns "
            f"{sorted(actual_component_set)} in '{filepath}'."
        )
    return None


def load_iaga2002_station_data(
    filepath: str | Path,
    station_code: str,
    *,
    silent: bool = False,
    log: LogCallback = None,
) -> Optional[pd.DataFrame]:
    """Load IAGA2002 data and normalize it to XYZ columns."""
    try:
        station_prefix = str(station_code).upper()
        filepath = Path(filepath)
        header = _parse_iaga2002_header(filepath)
        if header is None:
            if not silent and log is not None:
                log(f"Format Error: Data header not found in '{filepath}'.")
            return None

        available_components = [
            token.replace(station_prefix, "")
            for token in header.header_content
            if token.startswith(station_prefix)
        ]
        if not available_components:
            if not silent and log is not None:
                log(
                    f"Format Error: No data columns found for station {station_prefix} in '{filepath}'."
                )
            return None

        metadata_error = _validate_iaga_component_metadata(
            station_prefix=station_prefix,
            filepath=filepath,
            actual_components=available_components,
            header=header,
        )
        if metadata_error is not None:
            if not silent and log is not None:
                log(metadata_error)
            return None

        use_cols = ["DATE", "TIME"] + [f"{station_prefix}{comp}" for comp in available_components]
        missing_data_flags = [99999.0, 88888.0, 99999.9, 88888.8]
        data = pd.read_csv(
            filepath,
            skiprows=header.header_line_num,
            header=0,
            sep=r"\s+",
            usecols=lambda column: str(column).strip() in use_cols,
            na_values=missing_data_flags,
        )

        has_xyz = all(component in available_components for component in ("X", "Y", "Z"))
        has_dh = all(component in available_components for component in ("D", "H"))
        if not has_xyz and has_dh:
            d_radians = np.radians(data[f"{station_prefix}D"] / 600.0)
            h_nt = data[f"{station_prefix}H"]
            data[f"{station_prefix}X"] = h_nt * np.cos(d_radians)
            data[f"{station_prefix}Y"] = h_nt * np.sin(d_radians)
            if "Z" not in available_components:
                if "I" not in available_components:
                    if not silent and log is not None:
                        log(
                            "Format Error: Cannot calculate Z component. "
                            f"Need 'Z' or 'I' column in '{filepath}'."
                        )
                    return None
                i_radians = np.radians(data[f"{station_prefix}I"] / 600.0)
                data[f"{station_prefix}Z"] = h_nt * np.tan(i_radians)
        elif not has_xyz:
            if not silent and log is not None:
                log(
                    "Format Error: Unsupported magnetic components "
                    f"{available_components} in '{filepath}'. Need XYZ or DHI/DHZ."
                )
            return None

        final_cols = [f"{station_prefix}X", f"{station_prefix}Y", f"{station_prefix}Z"]
        if not all(col in data.columns for col in final_cols):
            if not silent and log is not None:
                log(f"Format Error: Could not produce all required XYZ columns in '{filepath}'.")
            return None

        data["datetime"] = pd.to_datetime(data["DATE"] + " " + data["TIME"])
        data = data.set_index("datetime").drop(columns=["DATE", "TIME"])
        return data[final_cols]
    except Exception as exc:
        if not silent and log is not None:
            log(f"Error loading and processing magnetometer data from {filepath}: {exc}")
        return None


def compute_baseline_offset(
    measured_series: pd.Series,
    inductive_series: pd.Series,
    magnetostatic_series: pd.Series,
    *,
    baseline_start: datetime.datetime,
    baseline_end: datetime.datetime,
) -> float:
    """Return a robust pre-event baseline offset for one component.

    The offset is computed from the median level over the configured
    pre-event window. The measured baseline is aligned to the median of
    the two simulation baselines.
    """
    measured_base = measured_series.loc[baseline_start:baseline_end].dropna().median()
    inductive_base = inductive_series.loc[baseline_start:baseline_end].dropna().median()
    magnetostatic_base = magnetostatic_series.loc[baseline_start:baseline_end].dropna().median()
    simulation_base = np.nanmedian([inductive_base, magnetostatic_base])
    if pd.isna(measured_base) or not np.isfinite(simulation_base):
        return 0.0
    return float(measured_base - simulation_base)


def calculate_station_metrics(
    station_code: str,
    mag_df_full: Optional[pd.DataFrame],
    sim_df_full: pd.DataFrame,
    sim_steady_df_full: pd.DataFrame,
    *,
    simulation_start_time: datetime.datetime,
    baseline_window: TimeWindow = DEFAULT_BASELINE_WINDOW,
    analysis_window: TimeWindow = DEFAULT_ANALYSIS_WINDOW,
) -> StationMetrics:
    """Calculate baseline offsets, correlations, and RMSEs for one station."""
    nan_comp = {component: np.nan for component in _COMPONENTS}
    zero_baseline = {component: 0.0 for component in _COMPONENTS}
    nan_metrics = ComparisonMetrics(correlations=nan_comp.copy(), rmses=nan_comp.copy())

    if mag_df_full is None:
        return StationMetrics(
            inductive=nan_metrics,
            magnetostatic=nan_metrics,
            baseline_offsets=zero_baseline,
        )

    sim_date = simulation_start_time.date()
    baseline_start = datetime.datetime.combine(sim_date, baseline_window[0])
    baseline_end = datetime.datetime.combine(sim_date, baseline_window[1])
    analysis_start = datetime.datetime.combine(sim_date, analysis_window[0])
    analysis_end = datetime.datetime.combine(sim_date, analysis_window[1])

    station_prefix = str(station_code).upper()
    mag_cols = [f"{station_prefix}X", f"{station_prefix}Y", f"{station_prefix}Z"]
    mag_analysis = mag_df_full.loc[analysis_start:analysis_end]
    if mag_analysis.empty or mag_analysis[mag_cols].isnull().all().all():
        return StationMetrics(
            inductive=nan_metrics,
            magnetostatic=nan_metrics,
            baseline_offsets=zero_baseline,
        )

    sim_analysis = sim_df_full.loc[analysis_start:analysis_end]
    sim_steady_analysis = sim_steady_df_full.loc[analysis_start:analysis_end]
    baseline_offsets: dict[str, float] = {}
    inductive_corr: dict[str, float] = {}
    inductive_rmse: dict[str, float] = {}
    steady_corr: dict[str, float] = {}
    steady_rmse: dict[str, float] = {}

    for mag_col, component in zip(mag_cols, _COMPONENTS):
        baseline_diff = 0.0
        if mag_col in mag_df_full.columns and not mag_df_full[mag_col].dropna().empty:
            baseline_diff = compute_baseline_offset(
                mag_df_full[mag_col],
                sim_df_full[component],
                sim_steady_df_full[component],
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )
        baseline_offsets[component] = baseline_diff
        measured_baselined = mag_analysis[mag_col] - baseline_diff

        temp_ind = pd.DataFrame({"meas": measured_baselined, "sim": sim_analysis[component]}).dropna()
        if len(temp_ind) > 1:
            inductive_corr[component] = float(temp_ind["meas"].corr(temp_ind["sim"]))
            inductive_rmse[component] = float(
                np.sqrt(((temp_ind["meas"] - temp_ind["sim"]) ** 2).mean())
            )
        else:
            inductive_corr[component] = np.nan
            inductive_rmse[component] = np.nan

        temp_steady = pd.DataFrame({"meas": measured_baselined, "sim": sim_steady_analysis[component]}).dropna()
        if len(temp_steady) > 1:
            steady_corr[component] = float(temp_steady["meas"].corr(temp_steady["sim"]))
            steady_rmse[component] = float(
                np.sqrt(((temp_steady["meas"] - temp_steady["sim"]) ** 2).mean())
            )
        else:
            steady_corr[component] = np.nan
            steady_rmse[component] = np.nan

    return StationMetrics(
        inductive=ComparisonMetrics(correlations=inductive_corr, rmses=inductive_rmse),
        magnetostatic=ComparisonMetrics(correlations=steady_corr, rmses=steady_rmse),
        baseline_offsets=baseline_offsets,
    )
