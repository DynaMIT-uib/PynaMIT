"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.math.constants import RE
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.grid import Grid
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.simulation.mainfield import Mainfield
from pynamit.simulation.state import State
from pynamit.spherical_harmonics.sh_basis import SHBasis

FLOAT_ERROR_MARGIN = 1e-6  # Safety margin for floating point errors


class Dynamics(object):
    """Class for simulating dynamic MIT coupling.

    Manages the temporal evolution of the state of the ionosphere in
    response to field-aligned currents and neutral winds, giving rise to
    dynamic magnetosphere-ionosphere-thermosphere (MIT) coupling. Saves
    and loads simulation data to and from NetCDF files.

    Attributes
    ----------
    current_time : float
        Current simulation time in seconds.
    state : State
        Current state of the system.
    RI : float
        Radius of the ionosphere in meters.
    mainfield : Mainfield
        Main magnetic field model.
    """

    def __init__(
        self,
        dataset_filename_prefix="simulation",
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=RE + 110.0e3,
        mainfield_kind="dipole",
        mainfield_epoch=2020,
        mainfield_B0=None,
        FAC_integration_steps=np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11),
        ignore_PFAC=False,
        connect_hemispheres=False,
        latitude_boundary=50,
        ih_constraint_scaling=1e-5,
        PFAC_matrix=None,
        vector_jr=True,
        vector_conductance=True,
        vector_u=True,
        t0="2020-01-01 00:00:00",
        save_steady_states=True,
        integrator="euler",
    ):
        """Initialize the Dynamics class.

        Parameters
        ----------
        dataset_filename_prefix : str, optional
            Prefix for saving dataset files.
        Nmax : int, optional
            Maximum spherical harmonic degree.
        Mmax : int, optional
            Maximum spherical harmonic order.
        Ncs : int, optional
            Number of cubed sphere grid points per edge.
        RI : float, optional
            Ionospheric radius in meters.
        mainfield_kind : {'dipole', 'igrf',  'radial'}, optional
            Type of main magnetic field model.
        mainfield_epoch : int, optional
            Epoch year for main field model.
        mainfield_B0 : float, optional
            Main field strength.
        FAC_integration_steps : array-like, optional
            Integration radii for FAC poloidal field calculation.
        ignore_PFAC : bool, optional
            Whether to ignore FAC poloidal fields.
        connect_hemispheres : bool, optional
            Whether hemispheres are electrically connected.
        latitude_boundary : float, optional
            Simulation boundary latitude in degrees.
        ih_constraint_scaling : float, optional
            Scaling for interhemispheric coupling constraint.
        PFAC_matrix : array-like, optional
            Matrix giving polodial field of FACs.
        vector_jr : bool, optional
            Use vector representation for radial current.
        vector_conductance : bool, optional
            Use vector representation for conductances.
        vector_u : bool, optional
            Use vector representation for neutral wind.
        t0 : str, optional
            Start time in UTC format.
        save_steady_states : bool, optional
            Whether to calculate and save steady states.
        integrator : {'euler', 'exponential'}, optional
            Integrator type for time evolution.
        """
        self.dataset_filename_prefix = dataset_filename_prefix

        # Store setting arguments in xarray dataset
        settings = xr.Dataset(
            attrs={
                "Nmax": Nmax,
                "Mmax": Mmax,
                "Ncs": Ncs,
                "RI": RI,
                "latitude_boundary": latitude_boundary,
                "ignore_PFAC": int(ignore_PFAC),
                "connect_hemispheres": int(connect_hemispheres),
                "FAC_integration_steps": FAC_integration_steps,
                "ih_constraint_scaling": ih_constraint_scaling,
                "mainfield_kind": mainfield_kind,
                "mainfield_epoch": mainfield_epoch,
                "mainfield_B0": 0 if mainfield_B0 is None else mainfield_B0,
                "vector_jr": int(vector_jr),
                "vector_conductance": int(vector_conductance),
                "vector_u": int(vector_u),
                "t0": t0,
                "save_steady_states": int(save_steady_states),
                "integrator": integrator,
            }
        )

        # Overwrite settings with any settings existing on file.
        settings_on_file = (self.dataset_filename_prefix is not None) and os.path.exists(
            self.dataset_filename_prefix + "_settings.ncdf"
        )
        if settings_on_file:
            settings = xr.load_dataset(self.dataset_filename_prefix + "_settings.ncdf")

        # Load PFAC matrix if it exists on file.
        PFAC_matrix_on_file = (self.dataset_filename_prefix is not None) and os.path.exists(
            self.dataset_filename_prefix + "_PFAC_matrix.ncdf"
        )
        if PFAC_matrix_on_file:
            PFAC_matrix = xr.load_dataarray(self.dataset_filename_prefix + "_PFAC_matrix.ncdf")

        self.RI = settings.RI

        self.mainfield = Mainfield(
            kind=settings.mainfield_kind,
            epoch=settings.mainfield_epoch,
            hI=(settings.RI - RE) * 1e-3,
            B0=None if settings.mainfield_B0 == 0 else settings.mainfield_B0,
        )

        self.cs_basis = CSBasis(settings.Ncs)
        self.state_grid = Grid(theta=self.cs_basis.arr_theta, phi=self.cs_basis.arr_phi)

        self.bases = {
            "state": SHBasis(settings.Nmax, settings.Mmax),
            "steady_state": SHBasis(settings.Nmax, settings.Mmax),
            "jr": SHBasis(settings.Nmax, settings.Mmax),
            "conductance": SHBasis(settings.Nmax, settings.Mmax, Nmin=0),
            "u": SHBasis(settings.Nmax, settings.Mmax),
        }

        self.vector_storage = {
            "state": True,
            "steady_state": True,
            "jr": bool(settings.vector_jr),
            "conductance": bool(settings.vector_conductance),
            "u": bool(settings.vector_u),
        }

        self.vars = {
            "state": {"m_ind": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
            "steady_state": {"m_ind": "scalar"},
            "jr": {"jr": "scalar"},
            "conductance": {"etaP": "scalar", "etaH": "scalar"},
            "u": {"u": "tangential"},
        }

        self.basis_multiindices = {}
        for key in self.vars.keys():
            if self.vector_storage[key]:
                basis_index_arrays = self.bases[key].index_arrays
                basis_index_names = self.bases[key].index_names
            else:
                basis_index_arrays = [self.state_grid.theta, self.state_grid.phi]
                basis_index_names = ["theta", "phi"]

            if all(self.vars[key][var] == "scalar" for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    basis_index_arrays, names=basis_index_names
                )
            elif all(self.vars[key][var] == "tangential" for var in self.vars[key]):
                self.basis_multiindices[key] = pd.MultiIndex.from_arrays(
                    [np.tile(basis_index_arrays[i], 2) for i in range(len(basis_index_arrays))],
                    names=basis_index_names,
                )
            else:
                raise ValueError(
                    "Mixed scalar and tangential input (unsupported), or invalid input type"
                )

        # Initialize the state of the ionosphere.
        self.state = State(
            self.bases, self.mainfield, self.state_grid, settings, PFAC_matrix=PFAC_matrix
        )

        self.timeseries = {}
        self.load_timeseries()

        if "state" in self.timeseries.keys():
            # Select last data in state time series.
            self.current_time = np.max(self.timeseries["state"].time.values)
            self.select_timeseries_data("state")
        else:
            self.current_time = np.float64(0)
            self.state.set_model_coeffs(m_ind=np.zeros(self.bases["state"].index_length))

        if self.dataset_filename_prefix is None:
            self.dataset_filename_prefix = "simulation"

        if not settings_on_file:
            self.save_dataset(settings, "settings")
            print(
                "Saved settings to {}_settings.ncdf".format(self.dataset_filename_prefix),
                flush=True,
            )

        if not PFAC_matrix_on_file:
            self.save_dataset(self.state.T_to_Ve, "PFAC_matrix")
            print(
                "Saved PFAC matrix to {}_PFAC_matrix.ncdf".format(self.dataset_filename_prefix),
                flush=True,
            )

        self.save_steady_states = bool(settings.save_steady_states)

    def evolve_to_time(
        self,
        t,
        dt=np.float64(5e-4),
        sampling_step_interval=200,
        saving_sample_interval=10,
        quiet=False,
    ):
        """Evolve the system state to a specified time.

        Parameters
        ----------
        t : float
            Target time to evolve to in seconds.
        dt : float, optional
            Time step size in seconds.
        sampling_step_interval : int, optional
            Number of steps between samples.
        saving_sample_interval : int, optional
            Number of samples between saves.
        quiet : bool, optional
            Whether to suppress progress output.
        """
        # Logicals are True when time series differ from ones on disk.
        self.save_jr = False
        self.save_conductance = False
        self.save_u = False

        count = 0
        while True:
            self.select_input_data()

            self.state.update_E()

            if count % sampling_step_interval == 0:
                # Append current state to time series.
                self.state.update_m_imp()

                current_state_dataset = xr.Dataset(
                    data_vars={
                        self.bases["state"].short_name + "_m_ind": (
                            ["time", "i"],
                            self.state.m_ind.coeffs.reshape((1, -1)),
                        ),
                        self.bases["state"].short_name + "_m_imp": (
                            ["time", "i"],
                            self.state.m_imp.coeffs.reshape((1, -1)),
                        ),
                        self.bases["state"].short_name + "_Phi": (
                            ["time", "i"],
                            self.state.E.coeffs[0].reshape((1, -1)),
                        ),
                        self.bases["state"].short_name + "_W": (
                            ["time", "i"],
                            self.state.E.coeffs[1].reshape((1, -1)),
                        ),
                    },
                    coords=xr.Coordinates.from_pandas_multiindex(
                        self.basis_multiindices["state"], dim="i"
                    ).merge({"time": [self.current_time]}),
                )

                self.add_to_timeseries(current_state_dataset, "state")

                if self.save_steady_states:
                    # Calculate steady state and append to time series.
                    steady_state_m_ind = self.state.steady_state_m_ind()
                    steady_state_m_imp = self.state.calculate_m_imp(steady_state_m_ind)
                    steady_state_E_coeffs = self.state.calculate_E_coeffs(steady_state_m_ind)

                    current_steady_state_dataset = xr.Dataset(
                        data_vars={
                            self.bases["steady_state"].short_name + "_m_ind": (
                                ["i"],
                                steady_state_m_ind,
                            ),
                            self.bases["steady_state"].short_name + "_m_imp": (
                                ["i"],
                                steady_state_m_imp,
                            ),
                            self.bases["steady_state"].short_name + "_Phi": (
                                ["i"],
                                steady_state_E_coeffs[0],
                            ),
                            self.bases["steady_state"].short_name + "_W": (
                                ["i"],
                                steady_state_E_coeffs[1],
                            ),
                        },
                        coords=xr.Coordinates.from_pandas_multiindex(
                            self.basis_multiindices["steady_state"], dim="i"
                        ).merge({"time": [self.current_time]}),
                    )

                    self.add_to_timeseries(current_steady_state_dataset, "steady_state")

                # Save state and steady state time series.
                if count % (sampling_step_interval * saving_sample_interval) == 0:
                    self.save_timeseries("state")

                    if quiet:
                        pass
                    else:
                        print(
                            "Saved state at t = {:.2f} s".format(self.current_time),
                            end="\n" if self.save_steady_states else "\r",
                            flush=True,
                        )

                    if self.save_steady_states:
                        self.save_timeseries("steady_state")

                        if quiet:
                            pass
                        else:
                            print(
                                "Saved steady state at t = {:.2f} s".format(self.current_time),
                                end="\x1b[F",
                                flush=True,
                            )

            next_time = self.current_time + dt

            if next_time > t + FLOAT_ERROR_MARGIN:
                if quiet:
                    pass
                else:
                    print("\n\n")
                break

            self.state.evolve_m_ind(dt)
            self.current_time = next_time

            count += 1

    def impose_steady_state(self):
        """Calculate and impose a steady state solution."""
        self.select_input_data()

        self.state.set_model_coeffs(m_ind=self.state.steady_state_m_ind())

        self.state.update_m_imp()

    def set_FAC(
        self,
        FAC,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set field-aligned current (FAC) input.

        Converts FAC to radial current density by multiplying with the
        radial component of the main field, and sets the radial current
        density as input.

        Parameters
        ----------
        FAC : array-like
            Field-aligned current density in A/m².
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the FAC data.
        weights : array-like, optional
            Weights for the FAC data points.
        reg_lambda : float, optional
            Regularization parameter for the least squares solver.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        FAC_b_evaluator = FieldEvaluator(
            self.mainfield, Grid(lat=lat, lon=lon, theta=theta, phi=phi), self.RI
        )

        self.set_jr(
            FAC * FAC_b_evaluator.br,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_jr(
        self,
        jr,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set radial current density input.

        Parameters
        ----------
        jr : array-like
            Radial current density in A/m².
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the current data.
        weights : array-like, optional
            Weights for the current data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        input_data = {"jr": [np.atleast_2d(jr)]}

        self.set_input(
            "jr",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_conductance(
        self,
        Hall,
        Pedersen,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set Hall and Pedersen conductance values.

        Parameters
        ----------
        Hall : array-like
            Hall conductance.
        Pedersen : array-like
            Pedersen conductance.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the conductance data.
        weights : array-like, optional
            Weights for the conductance data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for the pseudo-inverse.
        """
        Hall = np.atleast_2d(Hall)
        Pedersen = np.atleast_2d(Pedersen)

        input_data = {"etaP": [np.empty_like(Pedersen)], "etaH": [np.empty_like(Hall)]}

        # Convert to resistivity.
        for i in range(max(input_data["etaP"][0].shape[0], 1)):
            input_data["etaP"][0][i] = Pedersen[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        for i in range(max(input_data["etaH"][0].shape[0], 1)):
            input_data["etaH"][0][i] = Hall[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        self.set_input(
            "conductance",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

    def set_u(
        self,
        u_theta,
        u_phi,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
    ):
        """Set neutral wind velocities.

        Parameters
        ----------
        u_theta : array-like
            Meridional (south) wind velocity in m/s.
        u_phi : array-like
            Zonal (east) wind velocity in m/s.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the wind data.
        weights : array-like, optional
            Weights for the wind data points.
        reg_lambda : float, optional
            Regularization parameter.
        """
        input_data = {"u": [np.atleast_2d(u_theta), np.atleast_2d(u_phi)]}

        self.set_input(
            "u",
            input_data,
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            time=time,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=1e-15,
        )

    def set_input(
        self,
        key,
        input_data,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set input data for the simulation.

        Parameters
        ----------
        key : str
            The type of input data ('jr', 'conductance', or 'u').
        input_data : dict
            Dictionary containing the input data arrays.
        lat, lon : array-like, optional
            Latitude/longitude coordinates in degrees.
        theta, phi : array-like, optional
            Colatitude/azimuth coordinates in degrees.
        time : array-like, optional
            Time points for the input data.
        weights : array-like, optional
            Weights for the input data points.
        reg_lambda : float, optional
            Regularization parameter.
        pinv_rtol : float, optional
            Relative tolerance for pseudo-inverse.

        Raises
        ------
        ValueError
            If neither (lat, lon) nor (theta, phi) coordinates are
            provided.
        """
        input_grid = Grid(lat=lat, lon=lon, theta=theta, phi=phi)

        if not hasattr(self, "input_basis_evaluators"):
            self.input_basis_evaluators = {}

        if not (
            key in self.input_basis_evaluators.keys()
            and np.allclose(
                input_grid.theta,
                self.input_basis_evaluators[key].grid.theta,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
            and np.allclose(
                input_grid.phi,
                self.input_basis_evaluators[key].grid.phi,
                rtol=0.0,
                atol=FLOAT_ERROR_MARGIN,
            )
        ):
            self.input_basis_evaluators[key] = BasisEvaluator(
                self.bases[key],
                input_grid,
                weights=weights,
                reg_lambda=reg_lambda,
                pinv_rtol=pinv_rtol,
            )

        if time is None:
            if any(
                [
                    input_data[var][component].shape[0] > 1
                    for var in input_data.keys()
                    for component in range(len(input_data[var]))
                ]
            ):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            time = self.current_time

        time = np.atleast_1d(time)

        for time_index in range(time.size):
            processed_data = {}

            for var in self.vars[key]:
                if self.vector_storage[key]:
                    grid_value_array = np.array(
                        [
                            input_data[var][component][time_index]
                            for component in range(len(input_data[var]))
                        ]
                    )
                    if len(input_data[var]) == 1:
                        grid_values = grid_value_array[0]
                    else:
                        grid_values = grid_value_array
                    vector = FieldExpansion(
                        self.bases[key],
                        basis_evaluator=self.input_basis_evaluators[key],
                        grid_values=grid_values,
                        field_type=self.vars[key][var],
                    )

                    processed_data[self.bases[key].short_name + "_" + var] = (
                        ["time", "i"],
                        vector.coeffs.reshape((1, -1)),
                    )

                else:
                    # Interpolate to state_grid
                    if self.vars[key][var] == "scalar":
                        interpolated_data = self.cs_basis.interpolate_scalar(
                            input_data[var][0][time_index],
                            input_grid.theta,
                            input_grid.phi,
                            self.state_grid.theta,
                            self.state_grid.phi,
                        )
                    elif self.vars[key][var] == "tangential":
                        interpolated_east, interpolated_north, _ = (
                            self.cs_basis.interpolate_vector_components(
                                input_data[var][1][time_index],
                                -input_data[var][0][time_index],
                                np.zeros_like(input_data[var][1][time_index]),
                                input_grid.theta,
                                input_grid.phi,
                                self.state_grid.theta,
                                self.state_grid.phi,
                            )
                        )
                        interpolated_data = np.hstack(
                            (-interpolated_north, interpolated_east)
                        )  # convert to theta, phi

                    processed_data["GRID_" + var] = (
                        ["time", "i"],
                        interpolated_data.reshape((1, -1)),
                    )

            dataset = xr.Dataset(
                data_vars=processed_data,
                coords=xr.Coordinates.from_pandas_multiindex(
                    self.basis_multiindices[key], dim="i"
                ).merge({"time": [time[time_index]]}),
            )

            self.add_to_timeseries(dataset, key)

        self.save_timeseries(key)

    def add_to_timeseries(self, dataset, key):
        """Add a dataset to the timeseries.

        Creates a new timeseries if one does not exist, otherwise
        concatenates the new data along the time dimension.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset containing the timeseries data.
        key : str
            The key identifying the type of data ('state', 'jr',
            'conductance', or 'u').
        """
        if key not in self.timeseries.keys():
            self.timeseries[key] = dataset.sortby("time")
        else:
            self.timeseries[key] = xr.concat(
                [self.timeseries[key].drop_sel(time=dataset.time, errors="ignore"), dataset],
                dim="time",
            ).sortby("time")

    def select_timeseries_data(self, key, interpolation=False):
        """Select time series data corresponding to the latest time.

        Parameters
        ----------
        key : str
            Key for the time series.
        interpolation : bool, optional
            Whether to use linear interpolation.

        Returns
        -------
        bool
            Whether the input data was selected.
        """
        input_selected = False

        if np.any(self.timeseries[key].time.values <= self.current_time + FLOAT_ERROR_MARGIN):
            if self.vector_storage[key]:
                short_name = self.bases[key].short_name
            else:
                short_name = "GRID"

            current_data = {}

            # Select latest data before the current time.
            dataset_before = self.timeseries[key].sel(
                time=[self.current_time + FLOAT_ERROR_MARGIN], method="ffill"
            )

            for var in self.vars[key]:
                current_data[var] = dataset_before[short_name + "_" + var].values.flatten()

            # If requested, add linear interpolation correction.
            if (
                interpolation
                and (key != "state")
                and np.any(
                    self.timeseries[key].time.values > self.current_time + FLOAT_ERROR_MARGIN
                )
            ):
                dataset_after = self.timeseries[key].sel(
                    time=[self.current_time + FLOAT_ERROR_MARGIN], method="bfill"
                )
                for var in self.vars[key]:
                    current_data[var] += (
                        (self.current_time - dataset_before.time.item())
                        / (dataset_after.time.item() - dataset_before.time.item())
                        * (
                            dataset_after[short_name + "_" + var].values.flatten()
                            - dataset_before[short_name + "_" + var].values.flatten()
                        )
                    )

        else:
            # No data is available from before the current time.
            return input_selected

        if not hasattr(self, "previous_data"):
            self.previous_data = {}

        # Update state if first call or difference with last selection.
        if not all([var in self.previous_data.keys() for var in self.vars[key]]) or (
            not all(
                [
                    np.allclose(
                        current_data[var],
                        self.previous_data[var],
                        rtol=FLOAT_ERROR_MARGIN,
                        atol=0.0,
                    )
                    for var in self.vars[key]
                ]
            )
        ):
            if key == "state":
                self.state.set_model_coeffs(m_ind=current_data["m_ind"])
                self.state.set_model_coeffs(m_imp=current_data["m_imp"])
                self.state.E = FieldExpansion(
                    basis=self.bases[key],
                    coeffs=np.array([current_data["Phi"], current_data["W"]]),
                    field_type="tangential",
                )

            if key == "jr":
                if self.vector_storage[key]:
                    jr = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=current_data["jr"],
                        field_type=self.vars[key]["jr"],
                    )
                else:
                    jr = current_data["jr"]

                self.state.set_jr(jr)

            elif key == "conductance":
                if self.vector_storage[key]:
                    etaP = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=current_data["etaP"],
                        field_type=self.vars[key]["etaP"],
                    )
                    etaH = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=current_data["etaH"],
                        field_type=self.vars[key]["etaH"],
                    )
                else:
                    etaP = current_data["etaP"]
                    etaH = current_data["etaH"]

                self.state.set_conductance(etaP, etaH)

            elif key == "u":
                if self.vector_storage[key]:
                    u = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=current_data["u"].reshape((2, -1)),
                        field_type=self.vars[key]["u"],
                    )
                else:
                    u = current_data["u"].reshape((2, -1))

                self.state.set_u(u)

            for var in self.vars[key]:
                self.previous_data[var] = current_data[var]

            input_selected = True

        return input_selected

    def select_input_data(self):
        """Select input data corresponding to the latest time."""
        timeseries_keys = list(self.timeseries.keys())

        if "state" in timeseries_keys:
            timeseries_keys.remove("state")
        if timeseries_keys is not None:
            for key in timeseries_keys:
                self.select_timeseries_data(key, interpolation=False)

    def save_dataset(self, dataset, name):
        """Save a dataset to NetCDF file.

        Parameters
        ----------
        dataset : xarray.Dataset or xarray.DataArray
            The dataset to save.
        name : str
            Name to use in the filename.
        """
        filename = self.dataset_filename_prefix + "_" + name + ".ncdf"

        try:
            dataset.to_netcdf(filename + ".tmp")
            os.rename(filename + ".tmp", filename)

        except Exception as e:
            if os.path.exists(filename + ".tmp"):
                os.remove(filename + ".tmp")
            raise e

    def load_dataset(self, name):
        """Load dataset from file.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Returns
        -------
        xarray.Dataset or None
            Loaded dataset, or None if the file does not exist.
        """
        filename = self.dataset_filename_prefix + "_" + name + ".ncdf"

        if os.path.exists(filename):
            return xr.load_dataset(filename)
        else:
            return None

    def save_timeseries(self, key):
        """Save a timeseries to NetCDF file.

        Parameters
        ----------
        key : str
            The key identifying which timeseries to save.
        """
        self.save_dataset(self.timeseries[key].reset_index("i"), key)

    def load_timeseries(self):
        """Load all time series that exist on file."""
        if self.dataset_filename_prefix is not None:
            for key in self.vars.keys():
                dataset = self.load_dataset(key)

                if dataset is not None:
                    if self.vector_storage[key]:
                        basis_index_names = self.bases[key].index_names
                    else:
                        basis_index_names = ["theta", "phi"]

                    basis_multiindex = pd.MultiIndex.from_arrays(
                        [
                            dataset[basis_index_names[i]].values
                            for i in range(len(basis_index_names))
                        ],
                        names=basis_index_names,
                    )
                    coords = xr.Coordinates.from_pandas_multiindex(
                        basis_multiindex, dim="i"
                    ).merge({"time": dataset.time.values})
                    self.timeseries[key] = dataset.drop_vars(basis_index_names).assign_coords(
                        coords
                    )

    def calculate_fd_curl_matrix(self, stencil_size=1, interpolation_points=4):
        """Calculate matrix that returns the radial curl.

        Calculate matrix that maps column vector of (theta, phi) vector
        to its radial curl, using finite differences.

        Parameters
        ----------
        stencil_size : int, optional
            Size of the finite difference stencil.
        interpolation_points : int, optional
            Number of interpolation points.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix that returns the radial curl.
        """
        Dxi, Deta = self.cs_basis.get_Diff(
            self.cs_basis.N, coordinate="both", Ns=stencil_size, Ni=interpolation_points, order=1
        )

        g11_scaled = sp.diags(self.cs_basis.g[:, 0, 0] / self.cs_basis.sqrt_detg)
        g12_scaled = sp.diags(self.cs_basis.g[:, 0, 1] / self.cs_basis.sqrt_detg)
        g22_scaled = sp.diags(self.cs_basis.g[:, 1, 1] / self.cs_basis.sqrt_detg)

        # Construct matrix that gives radial curl from (u1, u2).
        D_curlr_u1u2 = sp.hstack(
            (
                (Dxi.dot(g12_scaled) - Deta.dot(g11_scaled)),
                (Dxi.dot(g22_scaled) - Deta.dot(g12_scaled)),
            )
        )

        # Construct matrix that transforms (theta, phi) to (u1, u2).
        Ps_dense = self.cs_basis.get_Ps(
            self.cs_basis.arr_xi, self.cs_basis.arr_eta, block=self.cs_basis.arr_block
        )

        # Extract relevant elements, rearrange matrix to map from
        # (theta, phi) and not (east, north). Also include Q matrix
        # normalization factors from Yin et al. (2017).
        RI_cos_lat = self.RI * np.cos(np.deg2rad(self.state_grid.lat))
        Ps = sp.vstack(
            (
                sp.hstack(
                    (
                        sp.diags(-Ps_dense[:, 0, 1] / self.RI),
                        sp.diags(Ps_dense[:, 0, 0] / RI_cos_lat),
                    )
                ),
                sp.hstack(
                    (
                        sp.diags(-Ps_dense[:, 1, 1] / self.RI),
                        sp.diags(Ps_dense[:, 1, 0] / RI_cos_lat),
                    )
                ),
            )
        )

        return D_curlr_u1u2.dot(Ps)

    @property
    def fd_curl_matrix(self):
        """Matrix for finite difference curl calculation."""
        if not hasattr(self, "_fd_curl_matrix"):
            self._fd_curl_matrix = self.calculate_fd_curl_matrix()
        return self._fd_curl_matrix

    @property
    def sh_curl_matrix(self):
        """Matrix for spherical harmonic curl calculation.

        Matrix that gets divergence-free SH coefficients from vectors of
        (theta, phi)-components, constructed from Laplacian matrix and
        (inverse) evaluation matrices.
        """
        if not hasattr(self, "_sh_curl_matrix"):
            G_df_pinv = self.state.basis_evaluator.least_squares_helmholtz.ATWA_plus_R_pinv[
                self.state.basis.index_length :, :
            ]
            self._sh_curl_matrix = self.state.basis_evaluator.G.dot(
                self.state.basis.laplacian().reshape((-1, 1)) * G_df_pinv
            )
        return self._sh_curl_matrix
