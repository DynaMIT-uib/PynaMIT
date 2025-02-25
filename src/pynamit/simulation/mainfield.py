"""Mainfield module.

This module contains the Mainfield class for main magnetic field
representation.
"""

import ppigrf
import apexpy
import dipole
import numpy as np
from datetime import datetime
from pynamit.math.constants import RE


class Mainfield(object):
    """Class for representing the main magnetic field.

    Provides implementations of different magnetic field models,
    providing field components, coordinate mapping, and basis vectors.

    Available models:
    - dipole: Dipole magnetic field using IGRF coefficients for moment.
    - igrf: International Geomagnetic Reference Field in geocentric
      coordinates (with geodetic conversion ignored).
    - radial: Radial field lines with configurable magnitude.

    Attributes
    ----------
    kind : str
        Active field model type.
    dpl : dipole.Dipole
        Dipole field instance (if kind=='dipole').
    apx : apexpy.Apex
        Apex coordinate transformer (if kind=='igrf').

    Notes
    -----
    The models use different coordinate systems: dipole uses dipole
    coordinates, IGRF uses geocentric coordinates, and radial uses
    simple radial field lines. For IGRF, geodetic height is approximated
    as ``h = r - RE``.
    """

    def __init__(self, kind="dipole", epoch=2020, hI=0.0, B0=None):
        """Initialize a Mainfield instance.

        Parameters
        ----------
        kind : {'dipole', 'igrf', 'radial'}, optional
            Type of magnetic field model.
        epoch : int, optional
            Decimal year for field coefficients.
        hI : float, optional
            Ionospheric height in km.
        B0 : float, optional
            Field magnitude at ground for radial model in Tesla. If
            None, uses reference field for epoch.
        """
        if kind.lower() not in ["radial", "dipole", "igrf"]:
            raise ValueError("kind must be either radial, dipole or igrf")

        self.kind = kind.lower()

        # Define magnetic field and mapping functions for chosen model.
        if self.kind == "dipole":
            self.dpl = dipole.Dipole(epoch)

            def _Bfunc(r, theta, phi):
                Bn, Br = self.dpl.B(90 - theta, r * 1e-3)
                return (Br * 1e-9, -Bn * 1e-9, Bn * 0)

        elif self.kind == "igrf":
            self.apx = apexpy.Apex(epoch, refh=hI)
            epoch = datetime(epoch, 1, 1, 0, 0)

            def _Bfunc(r, theta, phi):
                Br, Btheta, Bphi = ppigrf.igrf_gc(r * 1e-3, theta, phi, epoch)
                return (Br * 1e-9, Btheta * 1e-9, Bphi * 1e-9)

        elif self.kind == "radial":
            # Use Dipole B0 as default.
            B0 = dipole.Dipole(epoch).B0 if B0 is None else B0

            def _Bfunc(r, theta, phi):
                r, theta, phi = np.broadcast_arrays(r, theta, phi)
                return ((RE / r) ** 2 * B0, r * 0, r * 0)

        else:
            raise Exception("impossible")

        self._Bfunc = _Bfunc

    def get_B(self, r, theta, phi):
        """Calculate magnetic field components.

        Parameters
        ----------
        r : array-like
            Radius in meters.
        theta : array-like
            Colatitude in degrees.
        phi : array-like
            Longitude in degrees.

        Returns
        -------
        Br : ndarray
            Radial component of the magnetic field.
        Btheta : ndarray
            Southward component of the magnetic field.
        Bphi : ndarray
            Eastward component of the magnetic field.

        Notes
        -----
        Arrays are broadcast to common shape. Output components are in
        spherical coordinate basis.
        """
        return self._Bfunc(r, theta, phi)

    def get_sinI(self, r, theta, phi):
        """Calculate sine of the inclination angle.

        Defined as the angle of the magnetic field with nadir.
        Broadcasting rules apply.

        Parameters
        ----------
        r : array-like
            Radius [m] of the points where the magnetic field is to be
            evaluated.
        theta : array-like
            Colatitude [deg] of the points where the magnetic field is
            to be evaluated.
        phi : array-like
            Longitude [deg] of the points where the magnetic field is
            to be evaluated.

        Returns
        -------
        sinI : array
            sin(inclination).
        """
        B = np.vstack(self.get_B(r, theta, phi))

        return -B[0] / np.linalg.norm(B, axis=0)

    def map_coords(self, r_dest, r, theta, phi):
        """Map coordinates along field lines.

        Maps points to new radius along magnetic field lines.

        Parameters
        ----------
        r_dest : float
            Destination radius in meters.
        r : array-like
            Starting radius in meters.
        theta : array-like
            Starting colatitude in degrees.
        phi : array-like
            Starting longitude in degrees.

        Returns
        -------
        theta_out : ndarray
            Mapped colatitude in degrees.
        phi_out : ndarray
            Mapped longitude in degrees.

        Notes
        -----
        Implementation differs by model type:
        - IGRF: Uses apex coordinates.
        - Dipole: Uses analytic dipole field line equation.
        - Radial: Angular coordinates unchanged.
        """
        r, theta, phi = np.broadcast_arrays(r, theta, phi)

        if self.kind == "radial":
            # Angular coordinates are kept the same.
            theta_out = theta
            phi_out = phi

        if self.kind == "dipole":
            # Map from r to r_dest for dipole field.
            hemisphere = np.sign(90 - theta)
            la_ = 90 - np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta)) * np.sqrt(r_dest / r)))
            theta_out = 90 - hemisphere * la_
            # Longitude is kept the same.
            phi_out = phi

        elif self.kind == "igrf":
            # Use apexpy to map along IGRF field lines.
            mlat, mlon = self.apx.geo2apex(90 - theta, phi, (r - RE) * 1e-3)
            lat_out, phi_out, _ = self.apx.apex2geo(mlat, mlon, (r_dest - RE) * 1e-3)
            theta_out = 90 - lat_out

        return (theta_out, phi_out)

    def conjugate_coordinates(self, r, theta, phi):
        """Find magnetically conjugate points.

        Calculates coordinates of magnetically connected points
        in opposite hemisphere.

        Parameters
        ----------
        r : array-like
            Radius in meters.
        theta : array-like
            Colatitude in degrees.
        phi : array-like
            Longitude in degrees.

        Returns
        -------
        theta_conj : ndarray
            Conjugate point colatitude in degrees.
        phi_conj : ndarray
            Conjugate point longitude in degrees.

        Raises
        ------
        ValueError
            If called with radial field model.

        Notes
        -----
        Implementation differs by model type:
        - IGRF: Uses apex coordinate transformations.
        - Dipole: Conjugate points are at (180°-`theta`, `phi`).
        - Radial: Not defined.
        """
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))

        if self.kind == "radial":
            raise ValueError("Conjugate coordinates do not exist with radial field lines")

        if self.kind == "dipole":
            theta_conj, phi_conj = (180 - theta, phi)

        if self.kind == "igrf":
            h = (r - RE) * 1e-3
            mlat, mlon = self.apx.geo2apex(90 - theta, phi, h)
            glat, phi_conj, _ = self.apx.apex2geo(-mlat, mlon, h)
            theta_conj = 90 - glat

        return (theta_conj, phi_conj)

    def basis_vectors(self, r, theta, phi):
        """Calculate apex coordinate basis vectors.

        Computes modified apex coordinate basis vectors defined in
        Richmond (1995).

        Parameters
        ----------
        r : array-like
            Radius in meters.
        theta : array-like
            Colatitude in degrees.
        phi : array-like
            Longitude in degrees.

        Returns
        -------
        d1, d2, d3 : ndarray
            Contravariant basis vectors, shape (3,N).
        e1, e2, e3 : ndarray
            Covariant basis vectors, shape (3,N).

        Notes
        -----
        Vector components are in spherical coordinates.

        Implementation differs by model type:
        - IGRF: Uses apex coordinate transformations.
        - Dipole: Uses analytic dipole expressions.
        - Radial: Uses simple orthonormal vectors.
        """
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        size = r.size
        d1 = np.empty((3, size))
        d2 = np.empty((3, size))
        d3 = np.empty((3, size))
        e1 = np.empty((3, size))
        e2 = np.empty((3, size))
        e3 = np.empty((3, size))

        if self.kind == "radial":
            e = np.vstack((np.ones(size), np.zeros(size), np.zeros(size)))
            n = np.vstack((np.zeros(size), np.ones(size), np.zeros(size)))
            u = np.vstack((np.zeros(size), np.zeros(size), np.ones(size)))
            d1, e1 = e
            d2, e2 = n * np.sign(self.B(RE, 0, 0)[0]) * (-1)
            d3, e3 = u * np.sign(self.B(RE, 0, 0)[0])

        if self.kind == "dipole":
            _d1, _d2, _d3, _e1, _e2, _e3 = self.dpl.get_apex_base_vectors(
                90 - theta, r * 1e-3, R=RE * 1e-3
            )
            # Transform vectors from east north up to r, theta phi.
            d1[0] = _d1[2]  # Radial
            d2[0] = _d2[2]  # Radial
            d3[0] = _d3[2]  # Radial
            e1[0] = _e1[2]  # Radial
            e2[0] = _e2[2]  # Radial
            e3[0] = _e3[2]  # Radial
            d1[1] = -_d1[1]  # Theta
            d2[1] = -_d2[1]  # Theta
            d3[1] = -_d3[1]  # Theta
            e1[1] = -_e1[1]  # Theta
            e2[1] = -_e2[1]  # Theta
            e3[1] = -_e3[1]  # Theta
            d1[2] = _d1[0]  # Phi
            d2[2] = _d2[0]  # Phi
            d3[2] = _d3[0]  # Phi
            e1[2] = _e1[0]  # Phi
            e2[2] = _e2[0]  # Phi
            e3[2] = _e3[0]  # Phi

        if self.kind == "igrf":
            _, _, _, _, _, _, _d1, _d2, _d3, _e1, _e2, _e3 = self.apx.basevectors_apex(
                90 - theta, phi, (r - RE) * 1e-3, coords="geo"
            )
            # Transform vectors from east north up to r, theta phi.
            d1[0] = _d1[2]  # Radial
            d1[1] = -_d1[1]  # Theta
            d1[2] = _d1[0]  # Phi
            d2[0] = _d2[2]  # Radial
            d2[1] = -_d2[1]  # Theta
            d2[2] = _d2[0]  # Phi
            d3[0] = _d3[2]  # Radial
            d3[1] = -_d3[1]  # Theta
            d3[2] = _d3[0]  # Phi
            e1[0] = _e1[2]  # Radial
            e1[1] = -_e1[1]  # Theta
            e1[2] = _e1[0]  # Phi
            e2[0] = _e2[2]  # Radial
            e2[1] = -_e2[1]  # Theta
            e2[2] = _e2[0]  # Phi
            e3[0] = _e3[2]  # Radial
            e3[1] = -_e3[1]  # Theta
            e3[2] = _e3[0]  # Phi

        return (d1, d2, d3, e1, e2, e3)

    def dip_equator(self, phi, theta=90):
        """Calculate colatitude of given magnetic latitude at phi.

        Parameters
        ----------
        phi : array-like
            Longitude [deg] at which to calculate the dip equator.
        theta : float, optional
            Magnetic latitude.

        Returns
        -------
        array
            The co-latitude of the dip equator at the given longitude.
        """
        phi = np.array(phi) % 360

        if self.kind == "radial":
            print('dip_equator: Not defined for mainfield.kind=="radial"')
            return np.full_like(phi, np.nan)

        if self.kind == "dipole":
            return np.zeros_like(phi) + theta

        if self.kind == "igrf":
            mlon = np.linspace(0, 360, 360)
            # Calculate latitude of evenly spaced points.
            lat, lon, _ = self.apx.apex2geo(90 - theta, mlon, self.apx.refh)
            # Interpolate to phi.
            return (np.interp(phi.flatten(), lon % 360, 90 - lat, period=360)).reshape(phi.shape)
