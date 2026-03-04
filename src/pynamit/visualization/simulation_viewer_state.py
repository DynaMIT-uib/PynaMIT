"""Saved-run viewer state handling for ``SimulationViewer``."""

from __future__ import annotations

import datetime
import warnings

import apexpy
import numpy as np
from dipole import Dipole

from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.math.constants import RE
from pynamit.postprocess.grid_evaluation import decode_conductance_entry_to_grids
from pynamit.primitives.grid import Grid
from pynamit.simulation.data import SimulationData


class _SimulationViewerState:
    """Internal saved-run viewer state and snapshot access for ``SimulationViewer``."""

    def __init__(
        self,
        run_directory,
        t=0,
        Nlat=60,
        Nlon=100,
        NCS_plot=10,
        mlatlim=50,
        steady_state=True,
    ):
        self.simulation_data = SimulationData.from_directory(run_directory)
        self.datasets = self.simulation_data.datasets
        self.settings = self.simulation_data.settings
        if steady_state and not self.simulation_data.has_dataset("steady_state"):
            warnings.warn(
                f"Could not find steady-state data in {run_directory!r}.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.T_to_Ve = self.simulation_data.pfac_matrix
        self.mlatlim = mlatlim
        settings = self.settings
        self.RI = settings.RI
        self.mainfield = self.simulation_data.mainfield

        self.vector_cs_basis = CSBasis(NCS_plot)
        k, i, j = self.vector_cs_basis.get_gridpoints(NCS_plot)
        arr_xi = self.vector_cs_basis.xi(i[:, :-1, :-1] + 0.5, NCS_plot).flatten()
        arr_eta = self.vector_cs_basis.eta(j[:, :-1, :-1] + 0.5, NCS_plot).flatten()
        _, arr_theta, arr_phi = self.vector_cs_basis.cube2spherical(
            arr_xi, arr_eta, k[:, :-1, :-1].flatten(), deg=True
        )
        self.global_vector_grid = Grid(theta=arr_theta, lon=arr_phi)

        self.t0 = datetime.datetime.strptime(settings.t0, "%Y-%m-%d %H:%M:%S")
        self.dp = Dipole(self.t0.year)

        self.state_spec = self.simulation_data.solution_spec
        self.conductance_spec = self.simulation_data.get_storage_spec("conductance")

        self.grids = {}
        lat, lon = np.linspace(-89.9, 89.9, Nlat), np.linspace(-180, 180, Nlon)
        self.lat, self.lon = np.meshgrid(lat, lon)
        self.grids["global"] = Grid(lat=self.lat, lon=self.lon)
        self.grids["global_vector"] = self.global_vector_grid

        self.mlat, self.mlon = np.meshgrid(
            np.linspace(mlatlim, 89.9, Nlat // 2), np.linspace(-180, 180, Nlon)
        )
        if settings.mainfield_kind.lower() == "igrf":
            self.apx = apexpy.Apex(self.t0.year, refh=(settings.RI - RE) * 1e-3)
            self.lat_n, self.lon_n, _ = self.apx.apex2geo(
                self.mlat, self.mlon, (settings.RI - RE) * 1e-3
            )
            self.lat_s, self.lon_s, _ = self.apx.apex2geo(
                -self.mlat, self.mlon, (settings.RI - RE) * 1e-3
            )
            self.grids["north"] = Grid(lat=self.lat_n, lon=self.lon_n)
            self.grids["south"] = Grid(lat=self.lat_s, lon=self.lon_s)
        else:
            grid_polar = Grid(lat=self.mlat, lon=self.mlon)
            self.grids["north"] = grid_polar
            self.grids["south"] = grid_polar

        self.B_parameters_calculated = False
        self.operator_bundles = {}
        for region in ["global", "north", "south"]:
            self.operator_bundles[region] = self.simulation_data.get_poloidal_results_operators(
                basis=self.state_spec,
                grid=self.grids[region],
            )

        self.set_time(t)

    def _get_settings_scalar(self, name, default):
        """Return a scalar settings value from the loaded settings dataset."""
        if hasattr(self.settings, name):
            return float(getattr(self.settings, name))
        return float(default)

    def _get_conductance_entry_at_time(self):
        """Return the conductance representation stored at the current time."""
        entry = self.simulation_data.get_input_entry("conductance", self.t, interpolation=False)
        if entry is None:
            raise KeyError(f"No conductance entry is available at t={float(self.t):.2f}s.")
        return entry

    def _get_eta_on_grid(self, grid):
        """Decode the stored conductance representation to resistivity on ``grid``."""
        _, _, etaP, etaH = decode_conductance_entry_to_grids(
            self._get_conductance_entry_at_time(),
            self.conductance_spec,
            grid,
            target_shape=np.asarray(grid.lat).shape,
            sigma_floor=self._get_settings_scalar("conductance_interpolation_floor", 1e-3),
        )
        return etaP, etaH

    def set_time(self, t, steady_state=False):
        """Set current snapshot time in seconds."""
        self.t = t
        self.time = self.t0 + datetime.timedelta(seconds=t)

        state_entry = self.simulation_data.get_output_entry(
            "steady_state" if steady_state else "state",
            self.t,
            interpolation=False,
        )
        if state_entry is None:
            raise KeyError(
                f"No {'steady-state ' if steady_state else ''}state entry is available "
                f"at t={float(self.t):.2f}s."
            )
        u_entry = self.simulation_data.get_input_entry("u", self.t, interpolation=False)
        if u_entry is None:
            raise KeyError(f"No wind entry is available at t={float(self.t):.2f}s.")

        self.m_ind = np.asarray(state_entry["m_ind"])
        self.m_imp = np.asarray(state_entry["m_imp"])
        self.m_W = np.asarray(state_entry["W"]) * self.RI
        self.m_Phi = np.asarray(state_entry["Phi"]) * self.RI
        self.m_u = np.asarray(u_entry["u"]).reshape(2, -1)
        self.m_u_df, self.m_u_cf = np.split(self.m_u.flatten(), 2)

        if np.any(np.isnan(self.m_ind)):
            warnings.warn(
                f"Induced magnetic field coefficients at t={float(t):.2f}s contain NaNs.",
                RuntimeWarning,
                stacklevel=2,
            )

        return self
