"""
Spherical harmonic analysis helpers.

This module contains helpers for spherical harmonic analysis.

"""

import numpy as np

class SHKeys(object):
    """
    Container for ``n`` and ``m``, the indices of the terms in a spherical
    harmonic expansion.

    Container can be generated with::

        keys = SHKeys(Nmax, Mmax)

    ``keys`` will behave as a tuple of tuples, more or less.
    ``keys['n']`` will return a list of the ``n``'s.
    ``keys['m']`` will return a list of the ``m``'s.
    ``keys[3]`` will return the fourth ``(n,m)`` tuple.

    ``keys`` is also iterable.

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
    """
    Return number of coefficients in an expansion in real spherical
    harmonics.
    
    Toroidal magnetic potential is truncated at `NT`, `MT`.

    Poloidal magnetic potential is truncated at `NVi`, `MVi` for internal
    sources, and at `NVe`, `MVe` for external sources.

    """

    return len(SHKeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
           len(SHKeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
           len(SHKeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
           len(SHKeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
           len(SHKeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
           len(SHKeys(NVi, MVi).setNmin(1).MleN().Mge(1))


def legendre(Nmax, Mmax, theta, schmidtnormalize = True, keys = None):
    """
    Calculate associated Legendre function ``P`` and its derivative.

    Optional Schmidt semi-normalization.

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
    keys : SHKeys, optional
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
    for n in range(Nmax +1):
        for m in range(Nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, Nmax +1):
        for m in range(0, min([n + 1, Mmax + 1])):
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
        for n in range(1, Nmax + 1):
            for m in range(0, min([n + 1, Mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]


    if keys is None:
        return P, dP
    else:
        Pmat  = np.hstack(tuple(P[key] for key in keys))
        dPmat = np.hstack(tuple(dP[key] for key in keys)) 

        return np.hstack((Pmat, dPmat))