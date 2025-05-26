"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

import numpy as np
import xarray as xr
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.math.constants import RE
from pynamit.primitives.field_evaluator import FieldEvaluator
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

        # Specify input format and load input data.
        self.input_vars = {
            "jr": {"jr": "scalar"},
            "Br": {"Br": "scalar"},
            "conductance": {"etaP": "scalar", "etaH": "scalar"},
            "u": {"u": "tangential"},
        }

        self.input_storage_bases = {
            "jr": sh_basis_zero_removed,
            "Br": sh_basis_zero_removed,
            "conductance": sh_basis,
            "u": sh_basis_zero_removed,
        }

        self.input_timeseries = Timeseries(cs_basis, self.input_storage_bases, self.input_vars)
        self.input_timeseries.load_all(self.io)

        # Specify output format and load output data.
        self.output_vars = {
            "state": {"m_ind": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
            "steady_state": {"m_ind": "scalar", "m_imp": "scalar", "Phi": "scalar", "W": "scalar"},
        }

        self.output_storage_bases = {
            "state": sh_basis_zero_removed,
            "steady_state": sh_basis_zero_removed,
        }

        self.output_timeseries = Timeseries(cs_basis, self.output_storage_bases, self.output_vars)
        self.output_timeseries.load_all(self.io)

        self.interpolation_bases = {
            "jr": sh_basis_zero_removed if bool(self.settings.vector_jr) else cs_basis,
            "Br": sh_basis_zero_removed if bool(self.settings.vector_Br) else cs_basis,
            "conductance": sh_basis if bool(self.settings.vector_conductance) else cs_basis,
            "u": sh_basis_zero_removed if bool(self.settings.vector_u) else cs_basis,
        }

        self.mainfield = Mainfield(
            kind=self.settings.mainfield_kind,
            epoch=self.settings.mainfield_epoch,
            hI=(self.settings.RI - RE) * 1e-3,
            B0=None if self.settings.mainfield_B0 == 0 else self.settings.mainfield_B0,
        )

        # Initialize the state of the ionosphere, restarting from the
        # last state checkpoint if available.
        self.state = State(
            sh_basis_zero_removed,
            self.mainfield,
            cs_basis,
            self.settings,
            PFAC_matrix=PFAC_matrix_on_file,
        )

        if "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
        else:
            self.current_time = np.float64(0)

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
        steady_state_initialization=True,
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

        if "state" in self.output_timeseries.datasets.keys():
            self.current_time = np.max(self.output_timeseries.datasets["state"].time.values)
            inductive_m_ind = self.output_timeseries.get_entry_if_changed(
                "state", self.current_time, interpolation=False
            )
        else:
            if steady_state_initialization:
                self.state.update(self.input_timeseries, self.current_time)
                E_coeffs_noind, _ = self.state.calculate_noind_coeffs()
                inductive_m_ind = self.state.steady_state_m_ind(E_coeffs_noind)
            else:
                self.current_time = np.float64(0)
                inductive_m_ind = np.zeros(self.output_storage_bases["state"].index_length)

        while True:
            self.state.update(self.input_timeseries, self.current_time)

            E_coeffs_noind, m_imp_noind = self.state.calculate_noind_coeffs()

            if self.settings.integrator == "exponential" or (
                bool(self.settings.save_steady_states) and step % sampling_step_interval == 0
            ):
                steady_state_m_ind = self.state.steady_state_m_ind(E_coeffs_noind)
            else:
                steady_state_m_ind = None

            if step % sampling_step_interval == 0:
                self.add_state_to_timeseries("state", inductive_m_ind, E_coeffs_noind, m_imp_noind)

                if bool(self.settings.save_steady_states):
                    self.add_state_to_timeseries(
                        "steady_state", steady_state_m_ind, E_coeffs_noind, m_imp_noind
                    )

                # Save state and steady state time series.
                if step % (sampling_step_interval * saving_sample_interval) == 0:
                    self.output_timeseries.save("state", self.io)

                    if quiet:
                        pass
                    else:
                        print(
                            "Saved state at t = {:.2f} s".format(self.current_time),
                            end="\n" if bool(self.settings.save_steady_states) else "\r",
                            flush=True,
                        )

                    if bool(self.settings.save_steady_states):
                        self.output_timeseries.save("steady_state", self.io)

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

            inductive_m_ind = self.state.evolve_m_ind(
                inductive_m_ind, dt, E_coeffs_noind, steady_state_m_ind
            )
            self.current_time = next_time

            step += 1

    def add_state_to_timeseries(self, key, m_ind, E_coeffs_noind, m_imp_noind):
        """Add the current state to the time series.

        Parameters
        ----------
        key : str
            Key for the time series entry.
        m_ind : array-like
            Inductive magnetic field coefficients.
        E_coeffs_noind : tuple
            Electric field coefficients without induced effects.
        m_imp_noind : array-like
            Imposed magnetic field coefficients without induced effects.
        """
        E_coeffs_ind, m_imp_ind = self.state.calculate_ind_coeffs(m_ind)

        E_coeffs = E_coeffs_noind + E_coeffs_ind
        m_imp = m_imp_noind + m_imp_ind

        # Append current state to time series.
        state_data = {
            "SH_m_ind": m_ind,
            "SH_m_imp": m_imp,
            "SH_Phi": E_coeffs[0],
            "SH_W": E_coeffs[1],
        }

        self.output_timeseries.add_entry(key, state_data, time=self.current_time)

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

        self.input_timeseries.interpolate_and_add_entry(
            "jr",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["jr"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.input_timeseries.save("jr", self.io)

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

        self.input_timeseries.interpolate_and_add_entry(
            "Br",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["Br"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.input_timeseries.save("Br", self.io)

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

        self.input_timeseries.interpolate_and_add_entry(
            "conductance",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["conductance"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.input_timeseries.save("conductance", self.io)

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

        self.input_timeseries.interpolate_and_add_entry(
            "u",
            input_data,
            self.adapt_input_time(time, input_data),
            self.interpolation_bases["u"],
            lat=lat,
            lon=lon,
            theta=theta,
            phi=phi,
            weights=weights,
            reg_lambda=reg_lambda,
            pinv_rtol=pinv_rtol,
        )

        self.input_timeseries.save("u", self.io)

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
