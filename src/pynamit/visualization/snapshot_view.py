"""Internal snapshot rendering and movie export helpers for notebook-style views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from pynamit.visualization.map_plotting import (
    create_comparison_figure_axes,
    draw_comparison_colorbars,
    draw_comparison_field_sets,
    remove_artists,
)


@dataclass
class SnapshotRenderState:
    axes_groups: list[list[Any]]
    cbar_axes: list[Any]
    contours: list[Any]
    cbar1: Any = None
    cbar2: Any = None


class SnapshotMapRenderer:
    """Render snapshot comparison maps and export snapshot movies."""

    def __init__(
        self,
        *,
        global_lat: Any,
        global_lon: Any,
        north_mask: Any,
        south_mask: Any,
        plot_specs: dict[str, dict[str, Any]],
        diff_specs: dict[str, dict[str, Any]],
        get_time: Callable[[int], Any],
        format_time: Callable[[int], str],
        dipole: Any = None,
        global_projection: Optional[ccrs.Projection] = None,
        polar_minlat: float = 50.0,
    ) -> None:
        self.global_lat = global_lat
        self.global_lon = global_lon
        self.north_mask = north_mask
        self.south_mask = south_mask
        self.plot_specs = plot_specs
        self.diff_specs = diff_specs
        self.get_time = get_time
        self.format_time = format_time
        self.dipole = dipole
        self.global_projection = global_projection or ccrs.PlateCarree(central_longitude=0)
        self.polar_minlat = float(polar_minlat)

    @staticmethod
    def _variables_for(variable: str) -> list[str]:
        return ["Phi", "W"] if variable == "E_Field" else [variable]

    def draw(
        self,
        fig_handle: plt.Figure,
        *,
        plot_mode: str,
        idx: int,
        variable: str,
        fields_dict: dict[str, Any],
    ) -> SnapshotRenderState:
        plot_objects = create_comparison_figure_axes(
            plot_mode,
            existing_fig=fig_handle,
            global_projection=self.global_projection,
            polar_minlat=self.polar_minlat,
        )
        axes_groups = plot_objects["axes_groups"]
        cbar_axes = plot_objects["cbar_axes"]
        vars_to_plot = self._variables_for(variable)
        contours, main, diff = draw_comparison_field_sets(
            axes_groups,
            variables=vars_to_plot,
            fields_dict=fields_dict,
            plot_specs=self.plot_specs,
            diff_specs=self.diff_specs,
            global_lat=self.global_lat,
            global_lon=self.global_lon,
            north_mask=self.north_mask,
            south_mask=self.south_mask,
            time=self.get_time(idx),
            dipole=self.dipole,
        )
        cbar1, cbar2 = draw_comparison_colorbars(
            fig_handle,
            cbar_axes,
            main,
            diff,
            main_specs=[self.plot_specs[var] for var in vars_to_plot],
            diff_specs=[self.diff_specs[var] for var in vars_to_plot],
        )
        fig_handle.suptitle(f"Time: {self.format_time(idx)}", fontsize=16)
        return SnapshotRenderState(
            axes_groups=axes_groups,
            cbar_axes=cbar_axes,
            contours=contours[:],
            cbar1=cbar1,
            cbar2=cbar2,
        )

    def update(
        self,
        fig_handle: plt.Figure,
        render_state: SnapshotRenderState,
        *,
        idx: int,
        variable: str,
        fields_dict: dict[str, Any],
    ) -> None:
        remove_artists(render_state.contours)
        vars_to_plot = self._variables_for(variable)
        new_artists, _, _ = draw_comparison_field_sets(
            render_state.axes_groups,
            variables=vars_to_plot,
            fields_dict=fields_dict,
            plot_specs=self.plot_specs,
            diff_specs=self.diff_specs,
            global_lat=self.global_lat,
            global_lon=self.global_lon,
            north_mask=self.north_mask,
            south_mask=self.south_mask,
            time=self.get_time(idx),
            dipole=self.dipole,
        )
        render_state.contours[:] = new_artists
        fig_handle.suptitle(f"Time: {self.format_time(idx)}", fontsize=16)

    def save_movie(
        self,
        *,
        start_frame: int,
        end_frame: int,
        variable: str,
        plot_mode: str,
        fields_getter: Callable[[int], dict[str, Any]],
        filename: str | Path,
        fps: int = 10,
        dpi: int = 150,
        progress_factory: Optional[Callable[[int], Any]] = None,
    ) -> str:
        filename = str(filename)
        movie_fig = None
        pbar = None
        try:
            movie_fig_dict = create_comparison_figure_axes(
                plot_mode,
                global_projection=self.global_projection,
                polar_minlat=self.polar_minlat,
            )
            movie_fig = movie_fig_dict["fig"]
            movie_axes = movie_fig_dict["axes_groups"]
            movie_cbar_axes = movie_fig_dict["cbar_axes"]
            vars_to_plot = self._variables_for(variable)
            contours, main, diff = draw_comparison_field_sets(
                movie_axes,
                variables=vars_to_plot,
                fields_dict=fields_getter(start_frame),
                plot_specs=self.plot_specs,
                diff_specs=self.diff_specs,
                global_lat=self.global_lat,
                global_lon=self.global_lon,
                north_mask=self.north_mask,
                south_mask=self.south_mask,
                time=self.get_time(start_frame),
                dipole=self.dipole,
            )
            draw_comparison_colorbars(
                movie_fig,
                movie_cbar_axes,
                main,
                diff,
                main_specs=[self.plot_specs[var] for var in vars_to_plot],
                diff_specs=[self.diff_specs[var] for var in vars_to_plot],
            )
            time_text = movie_fig.suptitle(f"Time: {self.format_time(start_frame)}", fontsize=16)
            total_frames = int(end_frame - start_frame + 1)
            if progress_factory is not None:
                pbar = progress_factory(total_frames)

            def update(frame_idx: int) -> list[Any]:
                remove_artists(contours)
                new, _, _ = draw_comparison_field_sets(
                    movie_axes,
                    variables=vars_to_plot,
                    fields_dict=fields_getter(frame_idx),
                    plot_specs=self.plot_specs,
                    diff_specs=self.diff_specs,
                    global_lat=self.global_lat,
                    global_lon=self.global_lon,
                    north_mask=self.north_mask,
                    south_mask=self.south_mask,
                    time=self.get_time(frame_idx),
                    dipole=self.dipole,
                )
                contours.extend(new)
                time_text.set_text(f"Time: {self.format_time(frame_idx)}")
                if pbar is not None:
                    pbar.update(1)
                return contours + [time_text]

            ani = animation.FuncAnimation(
                movie_fig,
                update,
                frames=range(int(start_frame), int(end_frame) + 1),
                blit=False,
            )
            ani.save(filename, writer="ffmpeg", dpi=dpi, fps=fps)
            return filename
        finally:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            if movie_fig is not None:
                plt.close(movie_fig)
