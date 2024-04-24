"""
Tools that are useful for spherical harmonic analysis

Classes and functions in this module:

    SHkeys  : class to contain n and m - the indices of the spherical harmonic terms
    nterms  : function which calculates the number of terms in a real expansion of a poloidal (internal + external) and toroidal expansion
    legendre: calculate associated legendre functions - with option for Schmidt semi-normalization
    get_G   : calculate matrix for evaluating surface spherical harmonics at given grid


MIT License

Copyright (c) 2024 Karl M. Laundal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np

class SHkeys(object):
    """ container for n and m in spherical harmonics

        keys = SHkeys(Nmax, Mmax)

        keys will behave as a tuple of tuples, more or less
        keys['n'] will return a list of the n's
        keys['m'] will return a list of the m's
        keys[3] will return the fourth n,m tuple

        keys is also iterable

    """

    def __init__(self, Nmax, Mmax):
        keys = []
        for n in range(Nmax + 1):
            for m in range(Mmax + 1):
                keys.append((n, m))

        self.keys = tuple(keys)
        self.make_arrays()

    def __getitem__(self, index):
        if index == 'n':
            return [key[0] for key in self.keys]
        if index == 'm':
            return [key[1] for key in self.keys]

        return self.keys[index]

    def __iter__(self):
        for key in self.keys:
            yield key

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def __str__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def setNmin(self, nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= nmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def Mge(self, limit):
        """ set m >= limit  """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= limit])
        self.make_arrays()
        return self

    def NminusModd(self):
        """ remove keys if n - m is even """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """ remove keys if n - m is odd """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """ add negative m to the keys """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))
        
        self.keys = tuple(keys)
        self.make_arrays()
        
        return self


    def make_arrays(self):
        """ prepare arrays with shape ( 1, len(keys) )
            these are used when making G matrices
        """

        if len(self) > 0:
            self.m = np.array(self)[:, 1][np.newaxis, :]
            self.n = np.array(self)[:, 0][np.newaxis, :]
        else:
            self.m = np.array([])[np.newaxis, :]
            self.n = np.array([])[np.newaxis, :]



def nterms(NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
    """ return number of coefficients in an expansion in real spherical harmonics of
        toroidal magnetic potential truncated at NT, MT
        poloidal magnetic potential truncated at NVi, MVi for internal sources
        poloidal magnetic potential truncated at NVe, MVe for external sources
    """

    return len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(1))



def legendre(nmax, mmax, theta, schmidtnormalize = True, keys = None):
    """ Calculate associated Legendre function P and its derivative

        Algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz


        Parameters
        ----------
        nmax : int
            highest spherical harmonic degree
        mmax : int
            hightest spherical harmonic order
        theta : array, float
            colatitude in degrees (shape is not preserved)
        schmidtnormalize : bool, optional
            True if Schmidth seminormalization is wanted, False otherwise. Default True
        keys : SHkeys, optional
            If this parameter is set, an array will be returned instead of a dict. 
            The array will be (N, 2M), where N is the number of elements in `theta`, and 
            M is the number of keys. The first M columns represents a matrix of P values, 
            and the last M columns represent values of dP/dtheta

        Returns
        -------
        P : dict
            dictionary of Legendre function evalulated at theta. Dictionary keys are spherical harmonic
            wave number tuples (n, m), and values will have shape (N, 1), where N is number of 
            elements in `theta`. 
        dP : dict
            dictionary of Legendre function derivatives evaluated at theta. Dictionary keys are spherical
            harmonic wave number tuples (n, m), and values will have shape (N, 1), where N is number of 
            elements in theta. 
        PdP : array (only if keys != None)
            if keys != None, PdP is returned instaed of P and dP. PdP is an (N, 2M) array, where N is 
            the number of elements in `theta`, and M is the number of keys. The first M columns represents 
            a matrix of P values, and the last M columns represent values of dP/dtheta

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
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
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
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]


    if keys is None:
        return P, dP
    else:
        Pmat  = np.hstack(tuple(P[key] for key in keys))
        dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    
        return np.hstack((Pmat, dPmat))



def get_G(lat, lon, N, M, a = 6371.2, derivative = None, return_nm = False):
    """ Calculate matrix that evaluates surface spherical harmonics using the terms
        contained in shkeys, and at the locations defined by lat and lon

        Parameters
        ----------
        lat : array
            latitude in degrees. Must be broadcastable with lon
        lon : array
            longitude in degrees. Must be broadcastable with lat
        N: int
            maximum spherical harmonic degree
        M: int
            maximum spherical harmonic order
        a : float, optional
            Reference radius. Default is 6371.2
        derivative : string, optional
            Set to "phi" to get the matrix that gives the eastward gradient
            Set to "theta" to get the matrix that gives the southward gradient
            Default is None - the matrix gives surface SH (no derivative)


        Returns
        -------
        G : array
            N x M array, where N is the size inferred by broadcasting lon
            and lat, and M is the number of terms in the spherical harmonics
            inferred from shkeys. The cos terms are given first, and sin terms after
        
    """

    try: # broadcast lat and lon, and turn results into column vectors:
        lat, lon = np.broadcast_arrays(lat, lon)
        lat, lon = lat.flatten().reshape((-1, 1)), lon.flatten().reshape((-1, 1))
    except:
        raise Exception('get_G: could not brodcast lat and lon')

    ph, th = np.deg2rad(lon), np.deg2rad(90 - lat) # lon and colat in radians

    # make separate sets of spherical harmonic keys for cos and sin terms:
    cnm = SHkeys(N, M).setNmin(1).MleN()
    snm = SHkeys(N, M).setNmin(1).MleN().Mge(1)

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    PdP = legendre(N, M, 90 - lat.flatten(), keys = cnm)
    Pc, dPc = np.split(PdP, 2, axis = 1)
    Ps      =  Pc[: , cnm.m.flatten() != 0]
    dPs     = dPc[: , cnm.m.flatten() != 0]

    if derivative is None:
        Gc = a * Pc * np.cos(ph * cnm.m)
        Gs = a * Ps * np.sin(ph * snm.m)
    elif derivative == 'phi':
        Gc = -Pc * cnm.m * np.sin(ph * cnm.m) / np.sin(th)
        Gs =  Ps * snm.m * np.cos(ph * snm.m) / np.sin(th) 
    elif derivative == 'theta':
        Gc = dPc * np.cos(ph * cnm.m)
        Gs = dPs * np.sin(ph * snm.m)
    else:
        raise Exception(f'Invalid derivative "{derivative}". Expected: "phi", "theta", or None.')

    if return_nm:
        return np.hstack((Gc, Gs)), np.hstack((cnm.n.flatten(), snm.n.flatten())), np.hstack((cnm.m.flatten(), snm.m.flatten()))
    else:
        return np.hstack((Gc, Gs))

