"""Wigner 3j symbols and related functions."""

import numpy as np
import scipy.special
from scipy.special import gammaln
from functools import lru_cache


@lru_cache(maxsize=None)
def wigner_3j(j1, j2, j3, m1, m2, m3):
    """
    Compute the Wigner 3j symbol using the Racah formula.

    Parameters
    ----------
    j1, j2, j3 : int or float
        Angular momenta.
    m1, m2, m3 : int or float
        Magnetic quantum numbers.

    Returns
    -------
    float
        The value of the Wigner 3j symbol.
    """
    # Selection rules
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    
    if m1 + m2 + m3 != 0:
        return 0.0
    
    # Triangle inequality
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
        
    # Integer constraint on sum (for integer j, this is trivial, 
    # but for half-integer it matters. Here we assume integer input mostly)
    if (j1 + j2 + j3) % 1 != 0:
        return 0.0

    # Pre-compute factorials using gammaln (ln((n-1)!)) -> gammaln(n+1) = ln(n!)
    def lnfact(n):
        return gammaln(n + 1)
        
    term1 = lnfact(j1 + j2 - j3) + lnfact(j1 - j2 + j3) + lnfact(-j1 + j2 + j3) - lnfact(j1 + j2 + j3 + 1)
    term1 /= 2.0
    
    term2 = 0.5 * (
        lnfact(j1 + m1) + lnfact(j1 - m1) +
        lnfact(j2 + m2) + lnfact(j2 - m2) +
        lnfact(j3 + m3) + lnfact(j3 - m3)
    )
    
    # Summation range
    # max(0, j2-j3-m1, j1-j3+m2) <= k <= min(j1+j2-j3, j1-m1, j2+m2)
    k_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    k_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    
    sum_val = 0.0
    for k in range(int(k_min), int(k_max) + 1):
        denom = (
            lnfact(k) +
            lnfact(j1 + j2 - j3 - k) +
            lnfact(j1 - m1 - k) +
            lnfact(j2 + m2 - k) +
            lnfact(j3 - j2 + m1 + k) +
            lnfact(j3 - j1 - m2 + k)
        )
        term = (-1)**k * np.exp(-denom)
        sum_val += term
        
    # Combine
    log_coeff = term1 + term2
    result = (-1)**int(j1 - j2 - m3) * np.exp(log_coeff) * sum_val
    
    return result

@lru_cache(maxsize=None)
def wigner_6j(j1, j2, j3, J1, J2, J3):
    """
    Compute the Wigner 6j symbol { j1 j2 j3 }
                               { J1 J2 J3 }
    using Racah's formula.
    """
    if not _check_triangle(j1, j2, j3) or \
       not _check_triangle(j1, J2, J3) or \
       not _check_triangle(J1, j2, J3) or \
       not _check_triangle(J1, J2, j3):
        return 0.0
        
    def lnfact(n):
        return gammaln(n + 1)
        
    def delta(a, b, c):
        return lnfact(a+b-c) + lnfact(a-b+c) + lnfact(-a+b+c) - lnfact(a+b+c+1)
        
    triangles = delta(j1, j2, j3) + delta(j1, J2, J3) + delta(J1, j2, J3) + delta(J1, J2, j3)
    triangles /= 2.0
    
    k_min = max(j1+j2+j3, j1+J2+J3, J1+j2+J3, J1+J2+j3)
    k_max = min(j1+j2+J1+J2, j2+j3+J2+J3, j3+j1+J3+J1)
    
    sum_val = 0.0
    for k in range(int(k_min), int(k_max) + 1):
        denom = (
            lnfact(k - (j1+j2+j3)) +
            lnfact(k - (j1+J2+J3)) +
            lnfact(k - (J1+j2+J3)) +
            lnfact(k - (J1+J2+j3)) +
            lnfact(j1+j2+J1+J2 - k) +
            lnfact(j2+j3+J2+J3 - k) +
            lnfact(j3+j1+J3+J1 - k)
        )
        term = (-1)**k * lnfact(k+1) * np.exp(-denom)
        sum_val += term
        
    return np.exp(triangles) * sum_val

@lru_cache(maxsize=None)
def wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    """
    Compute the Wigner 9j symbol using summation over 6j symbols.

    Parameters
    ----------
    j1-j9 : int or float
        Angular momenta arranged as:
        { j1 j2 j3 }
        { j4 j5 j6 }
        { j7 j8 j9 }

    Returns
    -------
    float
        The value of the Wigner 9j symbol.
    """
    k_min = max(abs(j1-j9), abs(j2-j6), abs(j4-j8))
    k_max = min(j1+j9, j2+j6, j4+j8)

    total = 0.0
    for k in range(int(k_min), int(k_max) + 1):
        term = (2*k+1) * \
               wigner_6j(j1, j4, j7, j8, j9, k) * \
               wigner_6j(j2, j5, j8, j4, k, j6) * \
               wigner_6j(j3, j6, j9, k, j1, j2)
        total += (-1)**(2*k) * term

    return total

def _check_triangle(a, b, c):
    return abs(a - b) <= c <= a + b

def wigner_small_d(j, mp, m, beta):
    """
    Compute Wigner small-d matrix element d^j_{m', m}(beta).
    
    Using explicit summation formula (Wigner 1959).
    Valid for small j (stable up to j ~ 30).
    
    Parameters
    ----------
    j : float or int
        Angular momentum.
    mp : float or int
        m' (row index).
    m : float or int
        m (column index).
    beta : float or array-like
        Angle in radians.
        
    Returns
    -------
    d_val : float or ndarray
        Value of d^j_{m', m}(beta).
    """
    # Range of k summation
    # max(0, m - mp) <= k <= min(j + m, j - mp)
    k_min = max(0, m - mp)
    k_max = min(j + m, j - mp)
    
    if k_min > k_max:
        return 0.0
        
    # Pre-compute factorials
    def fact(n):
        return scipy.special.factorial(n)
        
    prefactor = np.sqrt(
        fact(j + mp) * fact(j - mp) * fact(j + m) * fact(j - m)
    )
    
    sum_val = 0.0
    
    # Cast beta to array for vectorization
    beta = np.asarray(beta)
    cos_half = np.cos(beta / 2.0)
    sin_half = np.sin(beta / 2.0)
    
    for k in range(int(k_min), int(k_max) + 1):
        num = (-1)**(k - m + mp)
        den = (
            fact(j + m - k) * 
            fact(k) * 
            fact(j - mp - k) * 
            fact(mp - m + k)
        )
        term = (num / den) * (cos_half**(2*j - mp + m - 2*k)) * (sin_half**(mp - m + 2*k))
        sum_val += term
        
    return prefactor * sum_val

