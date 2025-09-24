"""helpers.py - SHIndices and Schmidt quasi-normalization helpers."""

import numpy as np
import math
from typing import Iterable, Tuple

class SHIndices(object):
    """Container for (n,m) index pairs."""

    def __init__(self, Nmax: int, Mmax: int):
        index_pairs = []
        for n in range(Nmax + 1):
            for m in range(min(Mmax, n) + 1):
                index_pairs.append((n, m))
        self.index_pairs = tuple(index_pairs)
        self.make_arrays()

    def __getitem__(self, position):
        if position == "n":
            return [ip[0] for ip in self.index_pairs]
        if position == "m":
            return [ip[1] for ip in self.index_pairs]
        return self.index_pairs[position]

    def __iter__(self):
        for p in self.index_pairs:
            yield p

    def __len__(self):
        return len(self.index_pairs)

    def __repr__(self):
        return "".join(["n, m\n"] + [str(p)[1:-1] + "\n" for p in self.index_pairs])[:-1]

    def __str__(self):
        return self.__repr__()

    def set_Nmin(self, Nmin: int):
        self.index_pairs = tuple([p for p in self.index_pairs if p[0] >= Nmin])
        self.make_arrays()
        return self

    def set_Mmin(self, Mmin: int):
        self.index_pairs = tuple([p for p in self.index_pairs if abs(p[1]) >= Mmin])
        self.make_arrays()
        return self

    def make_arrays(self):
        if len(self.index_pairs) > 0:
            arr = np.array(self.index_pairs, dtype=int)
            self.n = arr[:, 0].reshape(1, -1)
            self.m = arr[:, 1].reshape(1, -1)
        else:
            self.n = np.array([]).reshape(1, -1)
            self.m = np.array([]).reshape(1, -1)
        # convenience map for fast lookups (avoid repeated list.index)
        self._index_map = {pair: i for i, pair in enumerate(self.index_pairs)}


def schmidt_quasi_normalization_factors(Nmax: int, Mmax: int):
    """
    Return a matrix of Schmidt quasi-normalization factors using an efficient
    O(n^2) recurrence relation.

    The factors are computed according to the geomagnetism convention
    (e.g., Langel, 1987).

    Parameters
    ----------
    Nmax : int
        Maximum degree.
    Mmax : int
        Maximum order.

    Returns
    -------
    S_matrix : ndarray, shape (Nmax+1, Mmax+1)
        Matrix of normalization factors where S_matrix[n, m] is the factor
        for the (n, m) pair.
    """
    S_matrix = np.zeros((Nmax + 1, Mmax + 1))
    S_matrix[0, 0] = 1.0

    for n in range(1, Nmax + 1):
        # Recurrence for m=0
        S_matrix[n, 0] = S_matrix[n - 1, 0] * (2.0 * n - 1.0) / n

        # Recurrence for m > 0
        for m in range(1, min(n, Mmax) + 1):
            factor_m_dep = 2.0 if m == 1 else 1.0
            factor = math.sqrt((n - m + 1.0) * factor_m_dep / (n + m))
            S_matrix[n, m] = S_matrix[n, m - 1] * factor

    return S_matrix