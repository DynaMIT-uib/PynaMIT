"""Internal notebook-support views for station-based ground comparisons."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Callable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynamit.postprocess.ground_station import (
    DEFAULT_ANALYSIS_WINDOW,
    DEFAULT_BASELINE_WINDOW,
    MagnetometerDataArchive,
    StationMetrics,
    TimeWindow,
    calculate_station_metrics,
)


@dataclass(frozen=True)
class StationComparisonPayload:
    station_code: str
    measured_full: Optional[pd.DataFrame]
    inductive_full: pd.DataFrame
    magnetostatic_full: pd.DataFrame
    measured_sliced: Optional[pd.DataFrame]
    inductive_sliced: pd.DataFrame
    magnetostatic_sliced: pd.DataFrame
    metrics: StationMetrics
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    baseline_window: TimeWindow
    analysis_window: TimeWindow
    error_message: Optional[str] = None

    def to_download_dataframe(self) -> pd.DataFrame:
        station_prefix = self.station_code.upper()
        index = (
            self.measured_sliced.index
            if self.measured_sliced is not None
            else self.inductive_sliced.index
        )
        data = pd.DataFrame(index=index)
        if self.measured_sliced is not None:
            data["Measured_X_nT"] = self.measured_sliced.get(f"{station_prefix}X")
            data["Measured_Y_nT"] = self.measured_sliced.get(f"{station_prefix}Y")
            data["Measured_Z_nT"] = self.measured_sliced.get(f"{station_prefix}Z")
        data["Inductive_North_nT"] = self.inductive_sliced["North"]
        data["Inductive_East_nT"] = self.inductive_sliced["East"]
        data["Inductive_Down_nT"] = self.inductive_sliced["Down"]
        data["Steady_North_nT"] = self.magnetostatic_sliced["North"]
        data["Steady_East_nT"] = self.magnetostatic_sliced["East"]
        data["Steady_Down_nT"] = self.magnetostatic_sliced["Down"]
        return data

    def download_header(self) -> str:
        analysis_start, analysis_end = self.analysis_window
        baseline_start, baseline_end = self.baseline_window
        header = (
            f"# Data for station: {self.station_code}\n"
            f"# Baseline window: {baseline_start.strftime('%H:%M')}–{baseline_end.strftime('%H:%M')} UTC\n"
            f"# Pearson correlation (r) and RMSE calculated for "
            f"{analysis_start.strftime('%H:%M')}–{analysis_end.strftime('%H:%M')} UTC\n"
        )
        for label, metrics in (("Inductive Sim", self.metrics.inductive), ("Magnetostatic Sim", self.metrics.magnetostatic)):
            header += f"# {label}:\n"
            for component in ("North", "East", "Down"):
                corr = metrics.correlations.get(component, np.nan)
                rmse = metrics.rmses.get(component, np.nan)
                header += f"# {component}: r={corr:.3f}, RMSE={rmse:.2f} nT\n"
        return header


class GroundStationComparisonBuilder:
    """Build station-comparison payloads and correlation reports for notebook views."""

    def __init__(
        self,
        *,
        simulation_start_time: datetime.datetime,
        sim_datetime_index: pd.DatetimeIndex,
        stations: pd.DataFrame,
        Br_inductive: np.ndarray,
        Bh_inductive: np.ndarray,
        Br_steady: np.ndarray,
        Bh_steady: np.ndarray,
        station_data_archive: MagnetometerDataArchive,
        baseline_window: TimeWindow = DEFAULT_BASELINE_WINDOW,
        analysis_window: TimeWindow = DEFAULT_ANALYSIS_WINDOW,
    ) -> None:
        self.simulation_start_time = simulation_start_time
        self.sim_datetime_index = pd.DatetimeIndex(pd.to_datetime(sim_datetime_index))
        self.stations = stations.reset_index(drop=True).copy()
        self.Br_inductive = np.asarray(Br_inductive, dtype=float)
        self.Bh_inductive = np.asarray(Bh_inductive, dtype=float)
        self.Br_steady = np.asarray(Br_steady, dtype=float)
        self.Bh_steady = np.asarray(Bh_steady, dtype=float)
        self.station_data_archive = station_data_archive
        self.baseline_window = baseline_window
        self.analysis_window = analysis_window
        self._station_index = {
            str(row["IAGA"]).upper(): int(idx) for idx, row in self.stations.iterrows()
        }
        self._sim_frame_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def station_codes(self) -> list[str]:
        return sorted(self._station_index)

    def plain_station_options(self) -> list[str]:
        return self.station_codes()

    def build_station_dropdown_options(
        self,
        *,
        end_time: datetime.datetime,
        progress_iter: Optional[Callable[[list[str]], Any]] = None,
    ) -> list[tuple[str, str]]:
        station_codes = self.station_codes()
        iterable = progress_iter(station_codes) if progress_iter is not None else station_codes
        options: list[tuple[str, str]] = []
        for station_code in iterable:
            status = self.station_data_archive.station_status(
                station_code,
                start_time=self.simulation_start_time,
                end_time=end_time,
            )
            options.append((f"{station_code} ({status})", station_code))
        return options

    def _get_station_index(self, station_code: str) -> int:
        try:
            return self._station_index[str(station_code).upper()]
        except KeyError as exc:
            raise KeyError(f"Station {station_code!r} not found.") from exc

    @staticmethod
    def _build_ground_dataframe(br_values: np.ndarray, bh_values: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "North": -np.asarray(bh_values[0], dtype=float) * 1e9,
                "East": np.asarray(bh_values[1], dtype=float) * 1e9,
                "Down": -np.asarray(br_values, dtype=float) * 1e9,
            },
            index=index,
        )

    def get_simulation_frames(self, station_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        station_code = str(station_code).upper()
        cached = self._sim_frame_cache.get(station_code)
        if cached is not None:
            return cached
        station_idx = self._get_station_index(station_code)
        inductive = self._build_ground_dataframe(
            self.Br_inductive[station_idx, :],
            self.Bh_inductive[:, station_idx, :],
            self.sim_datetime_index,
        )
        steady = self._build_ground_dataframe(
            self.Br_steady[station_idx, :],
            self.Bh_steady[:, station_idx, :],
            self.sim_datetime_index,
        )
        self._sim_frame_cache[station_code] = (inductive, steady)
        return inductive, steady

    def build_payload(
        self,
        station_code: str,
        *,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        download_if_missing: bool = True,
        silent: bool = False,
        log: Optional[Callable[[str], None]] = None,
    ) -> StationComparisonPayload:
        station_code = str(station_code).upper()
        inductive_full, steady_full = self.get_simulation_frames(station_code)
        measured_full = self.station_data_archive.get_station_data(
            station_code,
            download_if_missing=download_if_missing,
            silent=silent,
            log=log,
        )
        metrics = calculate_station_metrics(
            station_code,
            measured_full,
            inductive_full,
            steady_full,
            simulation_start_time=self.simulation_start_time,
            baseline_window=self.baseline_window,
            analysis_window=self.analysis_window,
        )
        if measured_full is None:
            return StationComparisonPayload(
                station_code=station_code,
                measured_full=None,
                inductive_full=inductive_full,
                magnetostatic_full=steady_full,
                measured_sliced=None,
                inductive_sliced=inductive_full.loc[start_time:end_time],
                magnetostatic_sliced=steady_full.loc[start_time:end_time],
                metrics=metrics,
                start_time=pd.Timestamp(start_time),
                end_time=pd.Timestamp(end_time),
                baseline_window=self.baseline_window,
                analysis_window=self.analysis_window,
                error_message=f"Could not load data for station {station_code}.",
            )
        return StationComparisonPayload(
            station_code=station_code,
            measured_full=measured_full,
            inductive_full=inductive_full,
            magnetostatic_full=steady_full,
            measured_sliced=measured_full.loc[start_time:end_time],
            inductive_sliced=inductive_full.loc[start_time:end_time],
            magnetostatic_sliced=steady_full.loc[start_time:end_time],
            metrics=metrics,
            start_time=pd.Timestamp(start_time),
            end_time=pd.Timestamp(end_time),
            baseline_window=self.baseline_window,
            analysis_window=self.analysis_window,
        )

    def build_correlation_report(
        self,
        *,
        download_if_missing: bool = True,
        silent: bool = True,
        log: Optional[Callable[[str], None]] = None,
        progress_iter: Optional[Callable[[list[str]], Any]] = None,
    ) -> pd.DataFrame:
        station_codes = self.station_codes()
        iterable = progress_iter(station_codes) if progress_iter is not None else station_codes
        results: list[dict[str, Any]] = []
        for station_code in iterable:
            inductive_full, steady_full = self.get_simulation_frames(station_code)
            measured_full = self.station_data_archive.get_station_data(
                station_code,
                download_if_missing=download_if_missing,
                silent=silent,
                log=log,
            )
            metrics = calculate_station_metrics(
                station_code,
                measured_full,
                inductive_full,
                steady_full,
                simulation_start_time=self.simulation_start_time,
                baseline_window=self.baseline_window,
                analysis_window=self.analysis_window,
            )
            station_idx = self._get_station_index(station_code)
            station_info = self.stations.loc[station_idx]
            results.append(
                {
                    "IAGA": station_code,
                    "Lat": float(station_info["GEOLAT"]),
                    "Lon": float(station_info["GEOLON"]),
                    "Corr_X_Ind": metrics.inductive.correlations["North"],
                    "RMSE_X_Ind_nT": metrics.inductive.rmses["North"],
                    "Corr_Y_Ind": metrics.inductive.correlations["East"],
                    "RMSE_Y_Ind_nT": metrics.inductive.rmses["East"],
                    "Corr_Z_Ind": metrics.inductive.correlations["Down"],
                    "RMSE_Z_Ind_nT": metrics.inductive.rmses["Down"],
                    "Corr_X_Steady": metrics.magnetostatic.correlations["North"],
                    "RMSE_X_Steady_nT": metrics.magnetostatic.rmses["North"],
                    "Corr_Y_Steady": metrics.magnetostatic.correlations["East"],
                    "RMSE_Y_Steady_nT": metrics.magnetostatic.rmses["East"],
                    "Corr_Z_Steady": metrics.magnetostatic.correlations["Down"],
                    "RMSE_Z_Steady_nT": metrics.magnetostatic.rmses["Down"],
                }
            )
        results_df = pd.DataFrame(results)
        subset_cols = [column for column in results_df.columns if "Corr" in column]
        results_df = results_df.dropna(how="all", subset=subset_cols)
        results_df["Mean_Corr_Ind"] = results_df[["Corr_X_Ind", "Corr_Y_Ind", "Corr_Z_Ind"]].mean(axis=1)
        results_df["Mean_Corr_Steady"] = results_df[
            ["Corr_X_Steady", "Corr_Y_Steady", "Corr_Z_Steady"]
        ].mean(axis=1)
        column_order = [
            "IAGA",
            "Lat",
            "Lon",
            "Mean_Corr_Ind",
            "Mean_Corr_Steady",
            "Corr_X_Ind",
            "RMSE_X_Ind_nT",
            "Corr_Y_Ind",
            "RMSE_Y_Ind_nT",
            "Corr_Z_Ind",
            "RMSE_Z_Ind_nT",
            "Corr_X_Steady",
            "RMSE_X_Steady_nT",
            "Corr_Y_Steady",
            "RMSE_Y_Steady_nT",
            "Corr_Z_Steady",
            "RMSE_Z_Steady_nT",
        ]
        results_df = results_df[column_order]
        results_df = results_df.sort_values(by="Mean_Corr_Ind", ascending=False).reset_index(drop=True)
        return results_df.round(3)


def draw_station_comparison_figure(fig_handle: Any, payload: StationComparisonPayload) -> None:
    fig_handle.clear()
    if payload.error_message is not None or payload.measured_sliced is None:
        ax = fig_handle.add_subplot(111)
        ax.text(0.5, 0.5, payload.error_message or "No station data available.", ha="center", va="center", color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    fig_handle.set_size_inches(12, 8)
    axes = fig_handle.subplots(nrows=3, ncols=1, sharex=True)
    fig_handle.suptitle(f"Comparison at {payload.station_code} Station", fontsize=16)
    station_prefix = payload.station_code.upper()
    mag_cols = [f"{station_prefix}X", f"{station_prefix}Y", f"{station_prefix}Z"]
    sim_cols = ["North", "East", "Down"]
    titles = ["North (X)", "East (Y)", "Down (Z)"]

    for ax, mag_col, sim_col, title in zip(axes, mag_cols, sim_cols, titles):
        offset = payload.metrics.baseline_offsets[sim_col]
        ax.plot(
            payload.measured_sliced.index,
            payload.measured_sliced[mag_col] - offset,
            color="k",
            label="Measured (offset)",
        )
        ax.plot(
            payload.inductive_sliced.index,
            payload.inductive_sliced[sim_col],
            color="r",
            linestyle="--",
            label="Inductive Sim.",
        )
        ax.plot(
            payload.magnetostatic_sliced.index,
            payload.magnetostatic_sliced[sim_col],
            color="b",
            linestyle=":",
            label="Magnetostatic Sim.",
        )
        ax.set_title(title)
        ax.set_ylabel("B (nT)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")

    fig_handle.set_constrained_layout(True)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel(f"Time on {payload.start_time.strftime('%Y-%m-%d')}")
    for label in axes[-1].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    fig_handle.canvas.draw_idle()


def create_correlation_summary_figure(results_df: pd.DataFrame) -> plt.Figure:
    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
    ax_corr.scatter(
        results_df["Lat"],
        results_df["Mean_Corr_Ind"],
        color="r",
        alpha=0.7,
        label="Inductive",
    )
    ax_corr.scatter(
        results_df["Lat"],
        results_df["Mean_Corr_Steady"],
        color="b",
        alpha=0.7,
        label="Magnetostatic",
    )
    mean_corr_ind = float(results_df["Mean_Corr_Ind"].mean())
    mean_corr_steady = float(results_df["Mean_Corr_Steady"].mean())
    ax_corr.axhline(
        mean_corr_ind,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Inductive Mean (r={mean_corr_ind:.2f})",
    )
    ax_corr.axhline(
        mean_corr_steady,
        color="darkblue",
        linestyle="--",
        linewidth=2,
        label=f"Magnetostatic Mean (r={mean_corr_steady:.2f})",
    )
    ax_corr.set_xlabel("Geographic Latitude (degrees)", fontsize=12)
    ax_corr.set_ylabel("Mean Pearson Correlation (r)", fontsize=12)
    ax_corr.set_title("Simulation Mean Correlation vs. Station Latitude", fontsize=14)
    ax_corr.set_ylim(-1.1, 1.1)
    ax_corr.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax_corr.grid(True, linestyle=":", alpha=0.6)
    ax_corr.legend()
    return fig_corr
