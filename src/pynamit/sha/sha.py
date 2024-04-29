"""
Tools that are useful for spherical harmonic analysis.

Functions in this module:

- ``nterms``: function which calculates the number of terms in a real
  expansion of a poloidal (internal + external) and toroidal expansion.
- ``legendre``: calculate associated legendre functions - with option for
  Schmidt semi-normalization.
- ``get_G``: calculate matrix for evaluating surface spherical harmonics
  at given grid.

"""
import numpy as np
from pynamit.sha.shkeys import SHkeys

class sha(object):
    """ Class for spherical harmonic analysis.

    """

    def __init__(self, Nmax, Mmax):
        self.Nmax = Nmax
        self.Mmax = Mmax

        # make separate sets of spherical harmonic keys for cos and sin terms:
        self.cnm = SHkeys(self.Nmax, self.Mmax).setNmin(1).MleN()
        self.snm = SHkeys(self.Nmax, self.Mmax).setNmin(1).MleN().Mge(1)



    def nterms(self, NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
        """
        Return number of coefficients in an expansion in real spherical
        harmonics of toroidal magnetic potential truncated at `NT`, `MT`.
    
        Poloidal magnetic potential truncated at `NVi`, `MVi` for internal
        sources and at `NVe`, `MVe` for external sources.
    
        """
    
        return len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
               len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
               len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
               len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
               len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
               len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(1))



    def legendre(self, theta, schmidtnormalize = True, keys = None):
        """
        Calculate associated Legendre function ``P`` and its derivative.

        Algorithm from "Spacecraft Attitude Determination and Control" by
        James Richard Wertz.

        Parameters
        ----------
        nmax : int
            Highest spherical harmonic degree.
        mmax : int
            Highest spherical harmonic order.
        theta : array, float
            Colatitude in degrees (shape is not preserved).
        schmidtnormalize : bool, optional, default = True
            ``True`` if Schmidth seminormalization is wanted, ``False``
            otherwise.
        keys : SHkeys, optional
            If this parameter is set, an array will be returned instead of
            a dict. The array will be ``(N, 2M)``, where ``N`` is the
            number of elements in `theta`, and ``M`` is the number of
            keys. The first ``M`` columns represents a matrix of ``P``
            values, and the last ``M`` columns represent values of
            ``dP/dtheta``.

        Returns
        -------
        P : dict
            If ``keys is None``, the dictionary of Legendre function
            evalulated at `theta` is returned. Dictionary keys are
            spherical harmonic wave number tuples ``(n, m)``, and values
            will have shape ``(N, 1)``, where ``N`` is the number of
            elements in `theta`. 
        dP : dict
            If ``keys is None``, the dictionary of Legendre function
            derivatives evaluated at `theta` is returned. Dictionary keys
            are spherical harmonic wave number tuples ``(n, m)``, and
            values will have shape ``(N, 1)``, where ``N`` is number of
            elements in theta.
        PdP : array
            If ``keys is not None``, ``PdP`` is returned instead of `P`
            and `dP`. `PdP` is an ``(N, 2M)`` array, where ``N`` is the
            number of elements in `theta`, and ``M`` is the number of
            keys. The first ``M`` columns represent a matrix of ``P``
            values, and the last ``M`` columns represent values of
            ``dP/dtheta``.

        """

        theta = theta.flatten()[:, np.newaxis]

        P = {}
        dP = {}
        sinth = np.sin(np.deg2rad(theta))
        costh = np.cos(np.deg2rad(theta))

        if schmidtnormalize:
            S = {}
            S[0, 0] = 1.

        # initialize the functions:
        for n in range(self.Nmax +1):
            for m in range(self.Nmax + 1):
                P[n, m] = np.zeros_like(theta, dtype = np.float64)
                dP[n, m] = np.zeros_like(theta, dtype = np.float64)

        P[0, 0] = np.ones_like(theta, dtype = np.float64)
        for n in range(1, self.Nmax +1):
            for m in range(0, min([n + 1, self.Mmax + 1])):
                # do the legendre functions and derivatives
                if n == m:
                    P[n, n]  = sinth * P[n - 1, m - 1]
                    dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
                else:

                    if n == 1:
                        Knm = 0.
                        P[n, m]  = costh * P[n -1, m]
                        dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                    elif n > 1:
                        Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                        P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                        dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

                if schmidtnormalize:
                    # compute Schmidt normalization
                    if m == 0:
                        S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                    else:
                        S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


        if schmidtnormalize:
            # now apply Schmidt normalization
            for n in range(1, self.Nmax + 1):
                for m in range(0, min([n + 1, self.Mmax + 1])):
                    P[n, m]  *= S[n, m]
                    dP[n, m] *= S[n, m]


        if keys is None:
            return P, dP
        else:
            Pmat  = np.hstack(tuple(P[key] for key in keys))
            dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    
            return np.hstack((Pmat, dPmat))



    def get_G(self, grid, derivative = None, return_nm = False):
        """
        Calculate matrix that evaluates surface spherical harmonics using
        the terms contained in ``shkeys``, and at the locations defined by
        `lat` and `lon`.

        Parameters
        ----------
        lat : array
            Latitude in degrees. Must be broadcastable with `lon`.
        lon : array
            Longitude in degrees. Must be broadcastable with `lat`.
        N: int
            Maximum spherical harmonic degree.
        M: int
            Maximum spherical harmonic order.
        a : float, optional, default = 6371.2
            Reference radius.
        derivative : string, {None, 'phi', 'theta'}, default = None
            Set to 'phi' to get the matrix that gives the eastward
            gradient.
            Set to 'theta' to get the matrix that gives the southward
            gradient. Default gives surface SH (no derivative).

        Returns
        -------
        G : array
            ``N x M`` array, where ``N`` is the size inferred by
            broadcasting `lon` and `lat`, and ``M`` is the number of terms
            in the spherical harmonics inferred from ``shkeys``. The
            ``cos`` terms are given first, and ``sin`` terms after.
        
        """

        try: # broadcast lat and lon, and turn results into column vectors:
            lat, lon = np.broadcast_arrays(grid.get_lat(), grid.get_lon())
            lat, lon = grid.get_lat().flatten().reshape((-1, 1)), grid.get_lon().flatten().reshape((-1, 1))
        except ValueError:
            raise Exception('get_G: could not brodcast lat and lon')

        ph, th = np.deg2rad(lon), np.deg2rad(90 - lat) # lon and colat in radians

        # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
        PdP = self.legendre(90 - lat.flatten(), keys = self.cnm)
        Pc, dPc = np.split(PdP, 2, axis = 1)
        Ps      =  Pc[: , self.cnm.m.flatten() != 0]
        dPs     = dPc[: , self.cnm.m.flatten() != 0]

        if derivative is None:
            Gc = grid.get_RI() * Pc * np.cos(ph * self.cnm.m)
            Gs = grid.get_RI() * Ps * np.sin(ph * self.snm.m)
        elif derivative == 'phi':
            Gc = -Pc * self.cnm.m * np.sin(ph * self.cnm.m) / np.sin(th)
            Gs =  Ps * self.snm.m * np.cos(ph * self.snm.m) / np.sin(th) 
        elif derivative == 'theta':
            Gc = dPc * np.cos(ph * self.cnm.m)
            Gs = dPs * np.sin(ph * self.snm.m)
        else:
            raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

        if return_nm:
            return np.hstack((Gc, Gs)), np.hstack((self.cnm.n.flatten(), self.snm.n.flatten())), np.hstack((self.cnm.m.flatten(), self.snm.m.flatten()))
        else:
            return np.hstack((Gc, Gs))
