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
            for m in range(min(Mmax, n) + 1):
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

    def set_Nmin(self, Nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= Nmin])
        self.make_arrays()
        return self

    def set_Mmin(self, Mmin):
        """ set minimum |m|  """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= Mmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set |m| <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
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
        """
        Prepare array of n and m indices with shape ( 1, len(keys) ).

        """

        if len(self) > 0:
            self.n = np.array(self)[:, 0].reshape(1, -1)
            self.m = np.array(self)[:, 1].reshape(1, -1)
        else:
            self.n = np.array([]).reshape(1, -1)
            self.m = np.array([]).reshape(1, -1)


def schmidt_normalization_factors(nm_tuples):
    """
    Return vector of Schmidt semi-normalization factors for spherical
    harmonic terms with given indices.

    Parameters
    ----------
    nm_tuples : list
        List of tuples of spherical harmonic indices.

    Returns
    -------
    S : vector
        Vector of Schmidt normalization factors for the given spherical
        harmonic indices.

    """

    S = np.empty(len(nm_tuples), dtype = np.float64)

    # Calculate the Schmidt normalization factors
    S[0] = 1.
    for nm in range(1, len(nm_tuples)):
        n, m = nm_tuples[nm]
        if m == 0:
            S[nm] = S[nm_tuples.index((n - 1, 0))] * (2. * n - 1.) / n
        else:
            factor = np.sqrt((n - m + 1.) * (int(m == 1) + 1.) / (n + m))
            S[nm] = S[nm_tuples.index((n, m - 1))] * factor
    return S