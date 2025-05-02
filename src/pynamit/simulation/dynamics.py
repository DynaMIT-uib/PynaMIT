"""Dynamics module.

This module contains the Dynamics class for simulating dynamic MIT
coupling.
"""

import os
import numpy as np
import scipy.sparse as sp
import xarray as xr
from pynamit.cubed_sphere.cs_basis import CSBasis
from pynamit.math.constants import RE
from pynamit.primitives.field_evaluator import FieldEvaluator
from pynamit.primitives.grid import Grid
from pynamit.primitives.field_expansion import FieldExpansion
from pynamit.simulation.timeseries import Timeseries
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

        # Initialize the state of the ionosphere.
        self.state = State(
            self.bases, self.mainfield, self.state_grid, settings, PFAC_matrix=PFAC_matrix
        )

        self.timeseries = Timeseries(self.bases, self.state_grid, self.cs_basis, self.vars, self.vector_storage)

        # Load all timeseries on file.
        for key in self.vars.keys():
            self.timeseries.load(key)

        if "state" in self.timeseries.datasets.keys():
            # Select last data in state time series.
            self.current_time = np.max(self.timeseries.datasets["state"].time.values)
            self.set_state_variables("state")
        else:
            self.current_time = np.float64(0)
            self.state.set_model_coeffs(m_ind=np.zeros(self.bases["state"].index_length))

        if self.dataset_filename_prefix is None:
            self.dataset_filename_prefix = "simulation"

        self.timeseries.set_dataset_filename_prefix(self.dataset_filename_prefix)

        if not settings_on_file:
            self.timeseries.save_dataset(settings, "settings")
            print(
                "Saved settings to {}_settings.ncdf".format(self.dataset_filename_prefix),
                flush=True,
            )

        if not PFAC_matrix_on_file:
            self.timeseries.save_dataset(self.state.T_to_Ve, "PFAC_matrix")
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
            self.set_input_state_variables()

            self.state.update_E()

            if count % sampling_step_interval == 0:
                # Append current state to time series.
                self.state.update_m_imp()

                state_data = {
                    "m_ind": [self.state.m_ind.coeffs.reshape((1, -1))],
                    "m_imp": [self.state.m_imp.coeffs.reshape((1, -1))],
                    "Phi": [self.state.E.coeffs[0].reshape((1, -1))],
                    "W": [self.state.E.coeffs[1].reshape((1, -1))],
                }

                self.timeseries.add_coeffs("state", state_data, time=self.current_time)

                if self.save_steady_states:
                    # Calculate steady state and append to time series.
                    steady_state_m_ind = self.state.steady_state_m_ind()
                    steady_state_m_imp = self.state.calculate_m_imp(steady_state_m_ind)
                    steady_state_E_coeffs = self.state.calculate_E_coeffs(steady_state_m_ind)

                    steady_state_data = {
                        "m_ind": [steady_state_m_ind.reshape((1, -1))],
                        "m_imp": [steady_state_m_imp.reshape((1, -1))],
                        "Phi": [steady_state_E_coeffs[0].reshape((1, -1))],
                        "W": [steady_state_E_coeffs[1].reshape((1, -1))],
                    }

                    self.timeseries.add_coeffs("steady_state", steady_state_data, time=self.current_time)

                # Save state and steady state time series.
                if count % (sampling_step_interval * saving_sample_interval) == 0:
                    self.timeseries.save("state")

                    if quiet:
                        pass
                    else:
                        print(
                            "Saved state at t = {:.2f} s".format(self.current_time),
                            end="\n" if self.save_steady_states else "\r",
                            flush=True,
                        )

                    if self.save_steady_states:
                        self.timeseries.save("steady_state")

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
        self.set_input_state_variables()

        self.state.set_model_coeffs(m_ind=self.state.steady_state_m_ind())

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
        input_data = {"jr": [jr]}

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

        # Convert conductances to resistances for all time points.
        for i in range(max(input_data["etaP"][0].shape[0], 1)):
            input_data["etaP"][0][i] = Pedersen[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

        for i in range(max(input_data["etaH"][0].shape[0], 1)):
            input_data["etaH"][0][i] = Hall[i] / (Hall[i] ** 2 + Pedersen[i] ** 2)

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
        input_data = {"u": [u_theta, u_phi]}

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
            pinv_rtol=1e-15,
        )

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
            if any(
                [
                    np.atleast_2d(data[var][component]).shape[0] > 1
                    for var in data.keys()
                    for component in range(len(data[var]))
                ]
            ):
                raise ValueError(
                    "Time must be specified if the input data is given for multiple time values."
                )
            return self.current_time
        else:
            return time

    def set_state_variables(self, key, interpolation=False):
        """Set input data for the simulation.

        Parameters
        ----------
        key : {'state', 'jr', 'conductance', 'u'}
            The type of input data.
        updated_data : dict
            Dictionary containing the input data variables for the
            specified key.
        """
        if key == "state":
            updated_data = self.timeseries.get_updated_data(
                key, self.current_time, interpolation=False
            )
        else:
            updated_data = self.timeseries.get_updated_data(
                key, self.current_time, interpolation=interpolation
            )

        if updated_data is not None:
            if key == "state":
                self.state.set_model_coeffs(m_ind=updated_data["m_ind"])
                self.state.set_model_coeffs(m_imp=updated_data["m_imp"])

                self.state.E = FieldExpansion(
                    basis=self.bases[key],
                    coeffs=np.array([updated_data["Phi"], updated_data["W"]]),
                    field_type="tangential",
                )

            if key == "jr":
                if self.vector_storage[key]:
                    jr = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=updated_data["jr"],
                        field_type=self.vars[key]["jr"],
                    )
                else:
                    jr = updated_data["jr"]

                self.state.set_jr(jr)

            elif key == "conductance":
                if self.vector_storage[key]:
                    etaP = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=updated_data["etaP"],
                        field_type=self.vars[key]["etaP"],
                    )
                    etaH = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=updated_data["etaH"],
                        field_type=self.vars[key]["etaH"],
                    )
                else:
                    etaP = updated_data["etaP"]
                    etaH = updated_data["etaH"]

                self.state.set_conductance(etaP, etaH)

            elif key == "u":
                if self.vector_storage[key]:
                    u = FieldExpansion(
                        basis=self.bases[key],
                        coeffs=updated_data["u"].reshape((2, -1)),
                        field_type=self.vars[key]["u"],
                    )
                else:
                    u = updated_data["u"].reshape((2, -1))

                self.state.set_u(u)

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
