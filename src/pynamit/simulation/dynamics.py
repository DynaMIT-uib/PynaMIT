"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

import numpy as np
import scipy.sparse as sp
import xarray as xr
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.math.constants import RE
from pynamit.primitives.basis_evaluator import BasisEvaluator
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.primitives.grid import Grid
from pynamit.primitives.io import IO
from pynamit.simulation.mainfield import Mainfield
from pynamit.simulation.state import State
from pynamit.primitives.timeseries import Timeseries
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
        filename_prefix="simulation",
        Nmax=20,
        Mmax=20,
        Ncs=30,
        RI=RE + 110.0e3,
        RM=None,
        mainfield_kind="dipole",
        mainfield_epoch=2020,
        mainfield_B0=None,
        FAC_integration_steps=np.logspace(np.log10(RE + 110.0e3), np.log10(4 * RE), 11),
        ignore_PFAC=False,
        connect_hemispheres=False,
        latitude_boundary=50,
        ih_constraint_scaling=1e-5,
        vector_jr=True,
        vector_Br=True,
        vector_conductance=True,
        vector_u=True,
        t0="2020-01-01 00:00:00",
        save_steady_states=True,
        integrator="euler",
    ):
        """Initialize the Dynamics class.

        Parameters
        ----------
        filename_prefix : str, optional
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
        vector_jr : bool, optional
            Use vector representation for radial current.
        vector_Br : bool, optional
            Use vector representation for radial magnetic field
            component.
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
        # Store setting arguments in xarray dataset.
        self.settings = xr.Dataset(
            attrs={
                "Nmax": Nmax,
                "Mmax": Mmax,
                "Ncs": Ncs,
                "RI": RI,
                "RM": 0 if RM is None else RM,
                "latitude_boundary": latitude_boundary,
                "ignore_PFAC": int(ignore_PFAC),
                "connect_hemispheres": int(connect_hemispheres),
                "FAC_integration_steps": FAC_integration_steps,
                "ih_constraint_scaling": ih_constraint_scaling,
                "mainfield_kind": mainfield_kind,
                "mainfield_epoch": mainfield_epoch,
                "mainfield_B0": 0 if mainfield_B0 is None else mainfield_B0,
                "vector_jr": int(vector_jr),
                "vector_Br": int(vector_Br),
                "vector_conductance": int(vector_conductance),
                "vector_u": int(vector_u),
                "t0": t0,
                "save_steady_states": int(save_steady_states),
                "integrator": integrator,
            }
        )

        self.vars = {
            "state": {"m_ind": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
            "steady_state": {"m_ind": "scalar"},
            "jr": {"jr": "scalar"},
            "Br": {"Br": "scalar"},
            "conductance": {"etaP": "scalar", "etaH": "scalar"},
            "u": {"u": "tangential"},
        }

        self.vector_storage = {
            "state": True,
            "steady_state": True,
            "jr": bool(self.settings.vector_jr),
            "Br": bool(self.settings.vector_Br),
            "conductance": bool(self.settings.vector_conductance),
            "u": bool(self.settings.vector_u),
        }

        self.io = IO(filename_prefix)

        # Check if settings are consistent with previously saved runs.
        settings_on_file = self.io.load_dataset("settings", print_info=True)

        if settings_on_file is not None:
            if not self.settings.identical(settings_on_file):
                raise ValueError(
                    "Mismatch between Dynamics object arguments and settings on file."
                )

        PFAC_matrix_on_file = self.io.load_dataarray("PFAC_matrix", print_info=True)

        sh_basis = SHBasis(self.settings.Nmax, self.settings.Mmax, Nmin=0)
        sh_basis_zero_removed = SHBasis(self.settings.Nmax, self.settings.Mmax)

        cs_basis = CSBasis(self.settings.Ncs)

        interpolation_bases = {
            "jr": sh_basis_zero_removed if self.vector_storage["jr"] else cs_basis,
            "Br": sh_basis_zero_removed if self.vector_storage["Br"] else cs_basis,
            "conductance": sh_basis if self.vector_storage["conductance"] else cs_basis,
            "u": sh_basis_zero_removed if self.vector_storage["u"] else cs_basis,
        }

        self.storage_bases = {
            "state": sh_basis_zero_removed,
            "steady_state": sh_basis_zero_removed,
            "jr": sh_basis_zero_removed,
            "Br": sh_basis_zero_removed,
            "conductance": sh_basis,
            "u": sh_basis_zero_removed,
        }

        self.mainfield = Mainfield(
            kind=self.settings.mainfield_kind,
            epoch=self.settings.mainfield_epoch,
            hI=(self.settings.RI - RE) * 1e-3,
            B0=None if self.settings.mainfield_B0 == 0 else self.settings.mainfield_B0,
        )

        self.timeseries = Timeseries(
            interpolation_bases, cs_basis, self.storage_bases, self.vars, self.vector_storage
        )

        # Load all timeseries on file.
        for key in self.vars.keys():
            self.timeseries.load(key, self.io)

        # Initialize the state of the ionosphere, restarting from the last
        # state checkpoint if available.
        self.state = State(
            sh_basis_zero_removed,
            self.timeseries.storage_basis_evaluators,
            self.mainfield,
            cs_basis,
            self.settings,
            PFAC_matrix=PFAC_matrix_on_file,
        )

        if "state" in self.timeseries.datasets.keys():
            self.current_time = np.max(self.timeseries.datasets["state"].time.values)
            self.set_state_variables("state")
        else:
            self.current_time = np.float64(0)
            self.state.m_ind = FieldExpansion(
                basis=self.storage_bases["state"],
                coeffs=np.zeros(self.storage_bases["state"].index_length),
                field_type=self.vars["state"]["m_ind"],
            )

        # Store settings and PFAC matrix on file.
        if filename_prefix is None:
            self.io.update_filename_prefix("simulation")

        if settings_on_file is None:
            self.io.save_dataset(self.settings, "settings", print_info=True)

        if PFAC_matrix_on_file is None:
            self.io.save_dataarray(self.state.T_to_Ve, "PFAC_matrix", print_info=True)

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
        step = 0

        while True:
            self.set_input_state_variables()

            self.state.update_E()

            if step % sampling_step_interval == 0:
                # Append current state to time series.
                self.state.update_m_imp()

                state_data = {
                    "SH_m_ind": self.state.m_ind.coeffs,
                    "SH_m_imp": self.state.m_imp.coeffs,
                    "SH_Phi": self.state.E.coeffs[0],
                    "SH_W": self.state.E.coeffs[1],
                }

                self.timeseries.add_entry("state", state_data, time=self.current_time)

                if bool(self.settings.save_steady_states):
                    # Calculate steady state and append to time series.
                    steady_state_m_ind = self.state.steady_state_m_ind()
                    steady_state_m_imp = self.state.calculate_m_imp(steady_state_m_ind)
                    steady_state_E_coeffs = self.state.calculate_E_coeffs(steady_state_m_ind)

                    steady_state_data = {
                        "SH_m_ind": steady_state_m_ind,
                        "SH_m_imp": steady_state_m_imp,
                        "SH_Phi": steady_state_E_coeffs[0],
                        "SH_W": steady_state_E_coeffs[1],
                    }

                    self.timeseries.add_entry(
                        "steady_state", steady_state_data, time=self.current_time
                    )

                # Save state and steady state time series.
                if step % (sampling_step_interval * saving_sample_interval) == 0:
                    self.timeseries.save("state", self.io)

                    if quiet:
                        pass
                    else:
                        print(
                            "Saved state at t = {:.2f} s".format(self.current_time),
                            end="\n" if bool(self.settings.save_steady_states) else "\r",
                            flush=True,
                        )

                    if bool(self.settings.save_steady_states):
                        self.timeseries.save("steady_state", self.io)

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

            step += 1

    def impose_steady_state(self):
        """Calculate and impose a steady state solution."""
        self.set_input_state_variables()

        self.state.m_ind = FieldExpansion(
            basis=self.storage_bases["state"],
            coeffs=self.state.steady_state_m_ind(),
            field_type=self.vars["steady_state"]["m_ind"],
        )

        self.state.update_m_imp()

    def set_input_state_variables(self):
        """Select input data corresponding to the latest time."""
        timeseries_keys = list(self.timeseries.datasets.keys())

        if "state" in timeseries_keys:
            timeseries_keys.remove("state")
        if timeseries_keys is not None:
            for key in timeseries_keys:
                self.set_state_variables(key, interpolation=False)

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
            self.mainfield, Grid(lat=lat, lon=lon, theta=theta, phi=phi), self.settings.RI
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
        input_data = {"jr": np.atleast_2d(jr)}

        self.timeseries.add_input(
            "jr",
            input_data,
            self.adapt_input_time(time, input_data),
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.timeseries.save("jr", self.io)

    def set_Br(
        self,
        Br,
        lat=None,
        lon=None,
        theta=None,
        phi=None,
        time=None,
        weights=None,
        reg_lambda=None,
        pinv_rtol=1e-15,
    ):
        """Set radial component of magnetic field input.

        Parameters
        ----------
        Br : array-like
            Radial component of magnetic field.
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
        if self.settings.RM == 0:
            raise ValueError("Br can only be set if magnetospheric radius (RM) is set.")

        input_data = {"Br": np.atleast_2d(Br)}

        self.timeseries.add_input(
            "Br",
            input_data,
            self.adapt_input_time(time, input_data),
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.timeseries.save("Br", self.io)

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

        input_data = {"etaP": np.empty_like(Pedersen), "etaH": np.empty_like(Hall)}

        # Convert conductances to resistances for all time points.
        for i in range(max(input_data["etaP"].shape[0], 1)):
            input_data["etaP"][i] = Pedersen[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        for i in range(max(input_data["etaH"].shape[0], 1)):
            input_data["etaH"][i] = Hall[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        self.timeseries.add_input(
            "conductance",
            input_data,
            self.adapt_input_time(time, input_data),
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.timeseries.save("conductance", self.io)

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
        pinv_rtol=1e-15,
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
        # If u_theta and u_phi, are 1D arrays, convert to 2D.
        input_data = {"u": np.array([np.atleast_2d(u_theta), np.atleast_2d(u_phi)])}
        # Reorder time to first dimension and component to second.
        input_data["u"] = np.moveaxis(input_data["u"], [0, 1], [1, 0])

        self.timeseries.add_input(
            "u",
            input_data,
            self.adapt_input_time(time, input_data),
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.timeseries.save("u", self.io)

    def adapt_input_time(self, time, data):
        """Adapt array of time values given with the input data.

        Parameters
        ----------
        time : array-like, optional
            Time values for the input data.
        data : dict
            Dictionary containing input data variables.

        Returns
        -------
        time : array-like
            Adapted time values.

        Notes
        -----
        If time is None, the current time is used.

        Raises
        ------
        ValueError
            If time is None and data is of a shape that suggests
            multiple time values.
        """
        if time is None:
            if any([data[var].shape[0] > 1 for var in data.keys()]):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return np.atleast_1d(self.current_time)
        else:
            return np.atleast_1d(time)

    def set_state_variables(self, key, interpolation=False):
        """Set input data for the simulation.

        Parameters
        ----------
        key : {'state', 'jr', 'Br', 'conductance', 'u'}
            The type of input data.
        updated_data : dict
            Dictionary containing the input data variables for the
            specified key.
        """

        updated_data = self.timeseries.get_entry_if_changed(
            key, self.current_time, interpolation=False if key == "state" else interpolation
        )

        if updated_data is not None:
            if key == "state":
                self.state.m_ind = FieldExpansion(
                    basis=self.storage_bases[key],
                    coeffs=updated_data["m_ind"],
                    field_type=self.vars[key]["m_ind"],
                )
                self.state.m_imp = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=updated_data["m_imp"],
                    field_type=self.vars[key]["m_imp"],
                )
                self.state.E = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=np.array([updated_data["Phi"], updated_data["W"]]),
                    field_type="tangential",
                )

            if key == "jr":
                self.state.jr = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=updated_data["jr"],
                    field_type=self.vars[key]["jr"],
                )

            if key == "Br":
                self.state.Br = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=updated_data["Br"],
                    field_type=self.vars[key]["Br"],
                )

            elif key == "conductance":
                etaP = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=updated_data["etaP"],
                    field_type=self.vars[key]["etaP"],
                )
                etaH = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=updated_data["etaH"],
                    field_type=self.vars[key]["etaH"],
                )

                self.state.update_matrices(etaP, etaH)

            elif key == "u":
                self.state.u = FieldExpansion(
                    self.storage_bases[key],
                    coeffs=updated_data["u"].reshape((2, -1)),
                    field_type=self.vars[key]["u"],
                )
