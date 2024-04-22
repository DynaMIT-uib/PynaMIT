""" main field models """

import ppigrf
import apexpy
import dipole
import numpy as np

RE = 6371.2e3 # Earth radius

class Mainfield(object):
    def __init__(self, kind = 'dipole', epoch = 2020., B0 = None):
        """
        Supported fields (kind kw):
        'dipole': 
            Dipole magnetic field, using IGRF coefficients to determine dipole
            moment. The functions will refer to *dipole coordinates*. Other parameters
            (FAC, conductanace, ...) must be given in the same coordinate system. 
        'igrf':
            International Geomagentic Reference Field, described in *geocentric* 
            coordinates. Other parameters (FAC, conductanace, ...) must be given in the
            same coordinate system. 
            NOTE: Conversion between geodetic and geocentric is ignored. geodetic height 
            is calculated as h = r - RE
        'radial':
            Radial field lines. You can specify the magnitude of B on ground through the
            B0 keyword. If not specified, the magnitude will be the (positive) dipole 
            reference field for the given epoch (and the epoch keyword is ignored)

        Parameters
        ----------
        kind: string, optional
            should be 'radial', 'dipole', or 'igrf'. Default is 'dipole'
        epoch: float, optional
            Define the epoch [decimal year] for the field model
        B0: float, optional
            Define the magnitude of the field on ground for kind == 'radial'
            The default is the reference field for epoch = 2020 (pointing upward)
        """

        if kind.lower() not in ['radial', 'dipole', 'igrf']:
            raise ValueError('kind must be either radial, dipole or igrf')

        self.kind = kind.lower()

        # define the magnetic field and mapping functions for the different options:
        if self.kind == 'dipole':
            self.dpl = dipole.Dipole(epoch)
            def _Bfunc(r, theta, phi):
                Bn, Br = self.dpl.B(90 - theta, r * 1e-3)
                return (Br, -Bn, Bn*0)

        elif self.kind == 'igrf':
            self.apx = apexpy.Apex(epoch, refh = 0)
            def _Bfunc(r, theta, phi):
                return ppigrf.igrf_gc(r * 1e-3, theta, phi, epoch)

        elif self.kind == 'radial':
            B0 = dipole.Dipole(epoch).B0 if B0 is None else B0 # use Dipole B0 as default
            def _Bfunc(r, theta, phi):
                r, theta, phi = np.broadcast_arrays(r, theta, phi)
                return ((RE / r)**2 * B0, r*0, r*0)

        else:
            raise Exception('impossible')

        self._Bfunc = _Bfunc


    def get_B(self, r, theta, phi):
        """ Calculate magnetic field vector [nT] at r [m], theta [deg], phi [deg]
        
        Broadcasting rules apply

        Parameters
        ----------
        r: array
            radius [m] of the points where the magnetic field is to be evaluated
        theta: array
            colatitude [deg] of the points where the magnetic field is to be evaluated
        phi: array
            longitude [deg] of the points where the magnetic field is to be evaluated

        Return
        ------
        Br : array
            Magnetic field [nT] in radial direction
        Btheta : array
            Magnetic field [nT] in theta (south) direction
        Bphi : array
            Magnetic field [nT] in eastward direction

        """

        return(self._Bfunc(r, theta, phi))


    def map_coords(self, r_dest, r, theta, phi):
        """ Map coordinates from r, theta, phi to a radius r_dest

        Broadcasting rules apply.

        Parameters
        ----------
        r_dest: float
            radius [m] to which we map the coordinates
        r: array
            radius [m] of the coordinates that shall be mapped to r_dest
        theta: array
            colatitude [deg] of the coordinates that shall be mapped to r_dest
        phi: array
            longitude [deg] of the coordinates that shall be mapped to r_dest

        Return
        ------
        theta_out: array
            colatitude [deg] of the input points when mapped to radius r_dest
        phi_out: array
            longitude [deg] of the input points when mapped to radius r_dest

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



    def basevectors(self, r, theta, phi):
        """ Get basevectors at r, theta, phi

        The basevectors are the apex basevectors as defined by Richmond 1995.
        For the three types of mainfield, we use different methods:
            - 'dipole': we use the dipole module, see documentation of that
                        module for full explanation
            - 'radial': the basevectors are orthonormal unit vectors
            - 'igrf': we use apexpy. NOTE: We treat theta as 90 - geodetic latitude
                      and r as RE + geodetic height. 


        Broadcasting rules apply, but output vectors will be (3, size), where 
        size is the size of the broadcast arrays

        Parameters
        ----------
        r: array
            radius [m] of the coordinates where we calcualte base vectors
        theta: array
            colatitude [deg] of the coordinates where we calcualte base vectors
        phi: array
            longitude [deg] of the coordinates where we calcualte base vectors

        Return
        ------
        d1, d2, d3, e1, e2, e3: arrays
            modifified apex base vectors, with the components referring to 
            east, north up
        """

        print('not tested!')
        r, theta, phi = map(np.ravel, np.broadcast_arrays(r, theta, phi))
        size = r.size

        if self.kind == 'radial':
            e = np.vstack((np.ones( size), np.zeros(size), np.zeros(size)))
            n = np.vstack((np.zeros(size), np.ones( size), np.zeros(size)))
            u = np.vstack((np.zeros(size), np.zeros(size), np.ones( size)))
            d1, e1 = e
            d2, e2 = n * np.sign(self.B(RE, 0, 0)[0]) * (-1)
            d3, e3 = u * np.sign(self.B(RE, 0, 0)[0])

        if self.kind == 'dipole':
            d1, d2, d3, e1, e2, e3 = self.dpl.get_apex_base_vectors(90 - theta, r * 1e-3, R = RE * 1e-3)

        if self.kind == 'igrf':
            d1, d2, d3, e1, e2, e3 = self.apx.basevectors_apex(90 - theta, phi, (r - RE) * 1e-3, coords='geo')

        return(d1, d2, d3, e1, e2, e3)



