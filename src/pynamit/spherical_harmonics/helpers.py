"""
Spherical harmonic analysis helpers.

This module contains helpers for spherical harmonic analysis.

Functions
---------
schmidt_normalization_factors
    Return vector of Schmidt semi-normalization factors for spherical harmonic terms.

Classes
-------
SHKeys
    Container for spherical harmonic expansion indices n and m.
"""
import numpy as np

class SHKeys(object):
    """Container for spherical harmonic expansion indices.
    
    A container for ``n`` and ``m``, the indices of the terms in a spherical
    harmonic expansion.

    Parameters
    ----------
    Nmax : int
        Maximum value for n index
    Mmax : int
        Maximum value for m index

    Attributes
    ----------
    keys : tuple
        Tuple of (n,m) index pairs
    n : ndarray
        Array of n indices, shape (1, len(keys))
    m : ndarray
        Array of m indices, shape (1, len(keys))

    Notes
    -----
    Container can be generated with::
        keys = SHKeys(Nmax, Mmax)

    The object provides dictionary-like access:
        - keys['n'] returns a list of n values
        - keys['m'] returns a list of m values
        - keys[i] returns the i-th (n,m) tuple
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
        """Set minimum n value.
        
        Parameters
        ----------
        Nmin : int
            Minimum value for n index
            
        Returns
        -------
        self : SHKeys
            Modified instance with filtered keys
        """
        self.keys = tuple([key for key in self.keys if key[0] >= Nmin])
        self.make_arrays()
        return self

    def set_Mmin(self, Mmin):
        """Set minimum absolute m value.
        
        Parameters
        ----------
        Mmin : int
            Minimum absolute value for m index
            
        Returns
        -------
        self : SHKeys
            Modified instance with filtered keys
        """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= Mmin])
        self.make_arrays()
        return self

    def MleN(self):
        """Filter keys to ensure |m| ≤ n.
        
        Returns
        -------
        self : SHKeys
            Modified instance with filtered keys
        """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def NminusModd(self):
        """Filter keys to keep only odd n-m differences.
        
        Returns
        -------
        self : SHKeys
            Modified instance with filtered keys where n-|m| is odd
        """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """Filter keys to keep only even n-m differences.
        
        Returns
        -------
        self : SHKeys
            Modified instance with filtered keys where n-|m| is even
        """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """Add negative m values to the keys.
        
        For each existing key with m≠0, adds a corresponding key with -m.
        
        Returns
        -------
        self : SHKeys
            Modified instance with added negative m keys
        """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))
        
        self.keys = tuple(keys)
        self.make_arrays()
        
        return self

    def make_arrays(self):
        """Create arrays of n and m indices.
        
        Creates n and m arrays with shape (1, len(keys)) for vectorized operations.
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