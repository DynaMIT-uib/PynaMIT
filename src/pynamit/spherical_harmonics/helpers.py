"""Helpers module.

This module contains helpers for spherical harmonic analysis, including
the SHIndices class for representing spherical harmonic indices and a
function for calculating Schmidt semi-normalization factors.
"""

import numpy as np


class SHIndices(object):
    """Class for representing spherical harmonic indices.

    A container for ``n`` and ``m``, the indices of the terms in a
    spherical harmonic expansion.

    Attributes
    ----------
    index_pairs : tuple
        Tuple of (n, m) index pairs.
    n : ndarray
        Array of n indices, shape (1, len(indices)).
    m : ndarray
        Array of m indices, shape (1, len(indices)).

    Notes
    -----
    Container can be generated with::
        indices = SHIndices(Nmax, Mmax)
    The object provides dictionary-like access:
        - indices['n'] returns a list of n values
        - indices['m'] returns a list of m values
        - indices[i] returns the i-th (n, m) index pair
    """

    def __init__(self, Nmax, Mmax):
        """Initialize the SHIndices instance.

        Parameters
        ----------
        Nmax : int
            Maximum value for n index.
        Mmax : int
            Maximum value for m index.
        """
        index_pairs = []
        for n in range(Nmax + 1):
            for m in range(min(Mmax, n) + 1):
                index_pairs.append((n, m))

        self.index_pairs = tuple(index_pairs)
        self.make_arrays()

    def __getitem__(self, position):
        """Return index pair at given position or list of ns or ms.

        Parameters
        ----------
        position : int, str
            Position of the index pair to return, or 'n' or 'm' to
            return a list of n or m indices.

        Returns
        -------
        index_pair : tuple, list
            The index pair at the given position or a list of n or m
            indices.
        """
        if position == "n":
            return [index_pair[0] for index_pair in self.index_pairs]
        if position == "m":
            return [index_pair[1] for index_pair in self.index_pairs]

        return self.index_pairs[position]

    def __iter__(self):
        """Return an iterator over the index pairs.

        Returns
        -------
        index_pair : tuple
            The next index pair in the container.
        """
        for index_pair in self.index_pairs:
            yield index_pair

    def __len__(self):
        """Return the number of index pairs in the container.

        Returns
        -------
        n_index_pairs : int
            Number of index pairs in the container.
        """
        return len(self.index_pairs)

    def __repr__(self):
        """Return official string representing the SHIndices instance.

        Returns
        -------
        str
            String representation of the SHIndices instance.
        """
        return "".join(
            ["n, m\n"]
            + [str(index_pair)[1:-1] + "\n" for index_pair in self.index_pairs]
        )[:-1]

    def __str__(self):
        """Return informal string representing the SHIndices instance.

        Returns
        -------
        str
            String representation of the SHIndices instance.
        """
        return "".join(
            ["n, m\n"]
            + [str(index_pair)[1:-1] + "\n" for index_pair in self.index_pairs]
        )[:-1]

    def set_Nmin(self, Nmin):
        """Set minimum n value.

        Parameters
        ----------
        Nmin : int
            Minimum value for n index.

        Returns
        -------
        self : SHIndices
            Modified instance with filtered indices.
        """
        self.index_pairs = tuple(
            [index_pair for index_pair in self.index_pairs if index_pair[0] >= Nmin]
        )
        self.make_arrays()
        return self

    def set_Mmin(self, Mmin):
        """Set minimum absolute m value.

        Parameters
        ----------
        Mmin : int
            Minimum absolute value for m index.

        Returns
        -------
        self : SHIndices
            Modified instance with filtered indices.
        """
        self.index_pairs = tuple(
            [
                index_pair
                for index_pair in self.index_pairs
                if abs(index_pair[1]) >= Mmin
            ]
        )
        self.make_arrays()
        return self

    def MleN(self):
        """Filter indices to ensure |m| ≤ n.

        Returns
        -------
        self : SHIndices
            Modified instance with filtered indices.
        """
        self.index_pairs = tuple(
            [
                index_pair
                for index_pair in self.index_pairs
                if abs(index_pair[1]) <= index_pair[0]
            ]
        )
        self.make_arrays()
        return self

    def NminusModd(self):
        """Filter indices to keep only odd n-m differences.

        Returns
        -------
        self : SHIndices
            Modified instance with filtered indices where n-|m| is odd.
        """
        self.index_pairs = tuple(
            [
                index_pair
                for index_pair in self.index_pairs
                if (index_pair[0] - abs(index_pair[1])) % 2 == 1
            ]
        )
        self.make_arrays()
        return self

    def NminusMeven(self):
        """Filter indices to keep only even n-m differences.

        Returns
        -------
        self : SHIndices
            Modified instance with filtered indices where n-|m| is even.
        """
        self.index_pairs = tuple(
            [
                index_pair
                for index_pair in self.index_pairs
                if (index_pair[0] - abs(index_pair[1])) % 2 == 0
            ]
        )
        self.make_arrays()
        return self

    def negative_m(self):
        """Add negative m values to the indices.

        For each existing index pair with m≠0, adds a corresponding
        index pair with -m.

        Returns
        -------
        self : SHIndices
            Modified instance with added index pairs with negative m.
        """
        index_pairs = []
        for index_pair in self.index_pairs:
            index_pairs.append(index_pair)
            if index_pair[1] != 0:
                index_pairs.append((index_pair[0], -index_pair[1]))

        self.index_pairs = tuple(index_pairs)
        self.make_arrays()

        return self

    def make_arrays(self):
        """Create arrays of n and m indices.

        Creates n and m arrays with shape (1, len(indices)) for
        vectorized operations.
        """
        if len(self) > 0:
            self.n = np.array(self)[:, 0].reshape(1, -1)
            self.m = np.array(self)[:, 1].reshape(1, -1)
        else:
            self.n = np.array([]).reshape(1, -1)
            self.m = np.array([]).reshape(1, -1)


def schmidt_normalization_factors(index_pairs):
    """Return Schmidt semi-normalization factors.

    Returns vector Schmidt semi-normalization factors for spherical
    harmonic terms with given indices.

    Parameters
    ----------
    index_pairs : list
        List of spherical harmonic index pairs.

    Returns
    -------
    S : vector
        Vector of Schmidt normalization factors for the given spherical
        harmonic index pairs.
    """
    S = np.empty(len(index_pairs), dtype=np.float64)

    # Calculate the Schmidt normalization factors.
    S[0] = 1.0
    for nm in range(1, len(index_pairs)):
        n, m = index_pairs[nm]
        if m == 0:
            S[nm] = S[index_pairs.index((n - 1, 0))] * (2.0 * n - 1.0) / n
        else:
            factor = np.sqrt((n - m + 1.0) * (int(m == 1) + 1.0) / (n + m))
            S[nm] = S[index_pairs.index((n, m - 1))] * factor
    return S
