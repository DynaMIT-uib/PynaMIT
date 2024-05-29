""" Main field models. """

import ppigrf
import apexpy
import dipole
import numpy as np
from datetime import datetime
from pynamit.constants import RE

class Mainfield(object):
    def __init__(self, kind = 'dipole', epoch = 2020, hI = 0., B0 = None):
        """
        Supported fields (with kind keywords):

        - 'dipole': Dipole magnetic field, using IGRF coefficients to
          determine dipole moment. The functions will refer to *dipole
          coordinates*. Other parameters (FAC, conductanace, ...) must be
          given in the same coordinate system.
        - 'igrf': International Geomagentic Reference Field, described
          in *geocentric*  coordinates. Other parameters (FAC,
          conductance, ...) must be given in the same coordinate system.
          NOTE: Conversion between geodetic and geocentric is ignored.
          Geodetic height is calculated as ``h = r - RE``.
        - 'radial': Radial field lines. You can specify the magnitude of
          ``B`` on ground through the `B0` keyword. If not specified, the
          magnitude will be the (positive) dipole reference field for the
          given epoch (and the epoch keyword is ignored).

        Parameters
        ----------
        kind : str, {'dipole', 'igrf', 'radial'}, default = 'dipole'
            The field model kind.
        epoch : int, optional
            The epoch [decimal year] for the field model.
        hI : float, optional
            Height of the ionosphere [km]
        B0 : float, optional
            The magnitude of the field on ground for ``kind == 'radial'``.
            The default is the reference field for ``epoch = 2020``
            (pointing upward).

        """

        if kind.lower() not in ['radial', 'dipole', 'igrf']:
            raise ValueError('kind must be either radial, dipole or igrf')

        self.kind = kind.lower()

        # define the magnetic field and mapping functions for the different options:
        if self.kind == 'dipole':
            self.dpl = dipole.Dipole(epoch)
            def _Bfunc(r, theta, phi):
                Bn, Br = self.dpl.B(90 - theta, r * 1e-3)
                return (Br * 1e-9, -Bn * 1e-9, Bn*0)

        elif self.kind == 'igrf':
            self.apx = apexpy.Apex(epoch, refh = hI)
            epoch = datetime(epoch, 1, 1, 0, 0)
            def _Bfunc(r, theta, phi):
                Br, Btheta, Bphi = ppigrf.igrf_gc(r * 1e-3, theta, phi, epoch)
                return (Br * 1e-9, Btheta * 1e-9, Bphi * 1e-9)

        elif self.kind == 'radial':
            B0 = dipole.Dipole(epoch).B0 if B0 is None else B0 # use Dipole B0 as default
            def _Bfunc(r, theta, phi):
                r, theta, phi = np.broadcast_arrays(r, theta, phi)
                return ((RE / r)**2 * B0, r*0, r*0)

        else:
            raise Exception('impossible')

        self._Bfunc = _Bfunc


    def get_B(self, r, theta, phi):
        """
        Calculate magnetic field vector [nT] at `r` [m], `theta` [deg],
        `phi` [deg].

        Broadcasting rules apply.

        Parameters
        ----------
        r: array
            Radius [m] of the points where the magnetic field is to be
            evaluated.
        theta: array
            Colatitude [deg] of the points where the magnetic field is to
            be evaluated.
        phi: array
            Longitude [deg] of the points where the magnetic field is to
            be evaluated.

        Return
        ------
        Br : array
            Magnetic field [nT] in radial direction.
        Btheta : array
            Magnetic field [nT] in `theta` (south) direction.
        Bphi : array
            Magnetic field [nT] in eastward direction.

        """

        return(self._Bfunc(r, theta, phi))


    def get_sinI(self, r, theta, phi):
        """ 
        Calculate sin inclination angle 

        Defined as the angle of the magnetic field with nadir

        Broadcasting rules apply.

        Parameters
        ----------
        r: array
            Radius [m] of the points where the magnetic field is to be
            evaluated.
        theta: array
            Colatitude [deg] of the points where the magnetic field is to
            be evaluated.
        phi: array
            Longitude [deg] of the points where the magnetic field is to
            be evaluated.

        Return
        ------
        sinI: array
            sin(inclination)

        """

        B = np.vstack(self.get_B(r, theta, phi))

        # return -Br/B0
        return(-B[0] / np.linalg.norm(B, axis = 0))


    def map_coords(self, r_dest, r, theta, phi):
        """ Map coordinates from `r`, `theta`, `phi` to a radius `r_dest`.

        Broadcasting rules apply.

        Parameters
        ----------
        r_dest: float
            Radius [m] to which we map the coordinates.
        r: array
            Radius [m] of the coordinates that shall be mapped to
            ``r_dest``.
        theta: array
            Colatitude [deg] of the coordinates that shall be mapped to
            ``r_dest``.
        phi: array
            Longitude [deg] of the coordinates that shall be mapped to
            ``r_dest``.

        Return
        ------
        theta_out: array
            Colatitude [deg] of the input points when mapped to radius
            ``r_dest``.
        phi_out: array
            Longitude [deg] of the input points when mapped to radius
            ``r_dest``.

        """

        r, theta, phi = np.broadcast_arrays(r, theta, phi)

        if self.kind == 'radial': # the angular coordinates are the same
            theta_out = theta
            phi_out = phi

        if self.kind == 'dipole': # Map from r to r_dest for dipole field:
            hemisphere = np.sign(90 - theta)
            la_ = 90 - np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta)) * np.sqrt(r_dest/r)))
            theta_out = 90 - hemisphere * la_
            phi_out = phi # longitude is the same

        elif self.kind == 'igrf': # Use apexpy to map along IGRF 
            mlat, mlon = self.apx.geo2apex(90 - theta, phi, (r - RE) * 1e-3)
            lat_out, phi_out, _ = self.apx.apex2geo(mlat, mlon, (r_dest - RE) * 1e-3)
            theta_out = 90 - lat_out

        return(theta_out, phi_out)

    def conjugate_coordinates(self, r, theta, phi):
        """ Calculate coordinates at magnetically conjugate points at radius `r`

        Parameters
        ----------
        r: array
            Radius [m] of original coordinates 
        theta: array
            Colatitude [deg] of original coordinates 
        phi: array
            Longitude [deg] of original coordinates 

        Return
        ------
        Parameters
        ----------
        theta_conj: array
            Colatitude [deg] of magnetically connected point in opposite hemisphere
        phi_conj: array
            Longitude [deg] of magnetically connected point in opposite hemisphere

        """

        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))

        if self.kind == 'radial':
            raise ValueError('Conjugate coordinates do not exist with radial field lines')

        if self.kind == 'dipole':
            theta_conj, phi_conj = 180 - theta, phi # assuming dipole coordinates are used

        if self.kind == 'igrf':
            h = (r - RE) * 1e-3
            mlat, mlon = self.apx.geo2apex(90 - theta, phi, h)
            glat, phi_conj, _ = self.apx.apex2geo(-mlat, mlon, h)
            theta_conj = 90 - glat

        return(theta_conj, phi_conj)



    def basevectors(self, r, theta, phi):
        """ Get basevectors at `r`, `theta`, `phi`.

        The basevectors are the apex basevectors as defined by Richmond
        1995. For the three types of mainfield, we use different methods:

        - 'dipole': we use the dipole module, see documentation of that
          module for full explanation.
        - 'radial': the basevectors are orthonormal unit vectors.
        - 'igrf': we use apexpy. NOTE: We treat `theta` as
          ``90 - geodetic latitude`` and `r` as ``RE + geodetic height``.


        Broadcasting rules apply, but output vectors will be
        ``(3, size)``, where ``size`` is the size of the broadcast arrays.

        Parameters
        ----------
        r: array
            Radius [m] of the coordinates where we calculate base vectors.
        theta: array
            Colatitude [deg] of the coordinates where we calculate base
            vectors.
        phi: array
            Longitude [deg] of the coordinates where we calculate base
            vectors.

        Return
        ------
        d1, d2, d3, e1, e2, e3: arrays
            Modifified apex base vectors, with the components referring to
            ``(r, theta, phi)``.

        """

        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        size = r.size
        d1 = np.empty((3, size))
        d2 = np.empty((3, size))
        d3 = np.empty((3, size))
        e1 = np.empty((3, size))
        e2 = np.empty((3, size))
        e3 = np.empty((3, size))



        if self.kind == 'radial':
            e = np.vstack((np.ones( size), np.zeros(size), np.zeros(size)))
            n = np.vstack((np.zeros(size), np.ones( size), np.zeros(size)))
            u = np.vstack((np.zeros(size), np.zeros(size), np.ones( size)))
            d1, e1 = e
            d2, e2 = n * np.sign(self.B(RE, 0, 0)[0]) * (-1)
            d3, e3 = u * np.sign(self.B(RE, 0, 0)[0])

        if self.kind == 'dipole':
            _d1, _d2, _d3, _e1, _e2, _e3 = self.dpl.get_apex_base_vectors(90 - theta, r * 1e-3, R = RE * 1e-3)
            # transform vectors from east north up to r, theta phi:
            d1[0] =  _d1[2] # radial
            d2[0] =  _d2[2] # radial
            d3[0] =  _d3[2] # radial
            e1[0] =  _e1[2] # radial
            e2[0] =  _e2[2] # radial
            e3[0] =  _e3[2] # radial
            d1[1] = -_d1[1] # theta
            d2[1] = -_d2[1] # theta
            d3[1] = -_d3[1] # theta
            e1[1] = -_e1[1] # theta
            e2[1] = -_e2[1] # theta
            e3[1] = -_e3[1] # theta
            d1[2] =  _d1[0] # phi
            d2[2] =  _d2[0] # phi
            d3[2] =  _d3[0] # phi
            e1[2] =  _e1[0] # phi
            e2[2] =  _e2[0] # phi
            e3[2] =  _e3[0] # phi


        if self.kind == 'igrf':
            _, _, _, _, _, _, _d1, _d2, _d3, _e1, _e2, _e3 = self.apx.basevectors_apex(90 - theta, phi, (r - RE) * 1e-3, coords='geo')
            # transform vectors from east north up to r, theta phi:
            d1[0] =  _d1[2] # radial
            d1[1] = -_d1[1] # theta
            d1[2] =  _d1[0] # phi
            d2[0] =  _d2[2] # radial
            d2[1] = -_d2[1] # theta
            d2[2] =  _d2[0] # phi
            d3[0] =  _d3[2] # radial
            d3[1] = -_d3[1] # theta
            d3[2] =  _d3[0] # phi
            e1[0] =  _e1[2] # radial
            e1[1] = -_e1[1] # theta
            e1[2] =  _e1[0] # phi
            e2[0] =  _e2[2] # radial
            e2[1] = -_e2[1] # theta
            e2[2] =  _e2[0] # phi
            e3[0] =  _e3[2] # radial
            e3[1] = -_e3[1] # theta
            e3[2] =  _e3[0] # phi


        return(d1, d2, d3, e1, e2, e3)


    def dip_equator(self, phi):
        """ Calculate the co-latitude of the dip equator at given phi """

        phi = np.array(phi) % 360

        if self.kind == 'radial':
            print('dip_equator: Not defined for mainfield.kind=="radial"')
            return np.full_like(phi, np.nan)

        if self.kind == 'dipole':
            return np.zeros_like(phi) + 90

        if self.kind == 'igrf':
            mlon = np.linspace(0, 360, 360)
            lat, lon, _ = self.apx.apex2geo(0, mlon, self.apx.refh) # lat of evenly spaced points
            # interpolate to phi:
            return( np.interp(phi.flatten(), lon % 360, 90 - lat, period = 360)).reshape(phi.shape) 








