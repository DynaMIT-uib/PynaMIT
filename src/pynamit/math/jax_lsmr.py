"""Small JAX LSMR implementation for internal least-squares solves.

Adapted from SciPy's ``lsmr`` implementation (BSD-3-Clause,
SciPy Developers) and the Fong/Saunders LSMR algorithm.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def lsmr(
    A: Any,
    b: Any,
    damp: float = 0.0,
    atol: float = 1e-6,
    btol: float = 1e-6,
    conlim: float = 1e8,
    maxiter: Optional[int] = None,
    show: bool = False,
    x0: Any = None,
) -> Tuple[Any, int, int, float, float, float, float, float]:
    """Solve ``min ||b - A x||`` using JAX array operations.

    This mirrors the core SciPy LSMR iteration for internal use with
    ``LinearMap`` objects. The iteration loop is intentionally
    controlled from Python so arbitrary map implementations can be used.
    """
    del show

    import jax.numpy as jnp

    m, n = A.shape
    if maxiter is None:
        maxiter = min(m, n)
    atol = 1e-6 if atol is None else atol
    btol = 1e-6 if btol is None else btol
    conlim = 1e8 if conlim is None else conlim

    b = jnp.atleast_1d(b).reshape(m)
    dtype = jnp.result_type(b, 1.0 if x0 is None else x0)
    u = b.astype(dtype)
    normb = _norm(u)

    if x0 is None:
        x = jnp.zeros(n, dtype=dtype)
        beta = normb
    else:
        x = jnp.atleast_1d(jnp.asarray(x0, dtype=dtype)).reshape(n)
        u = u - A.matvec(x)
        beta = _norm(u)

    if beta > 0:
        u = u / beta
        v = A.rmatvec(u)
        alpha = _norm(v)
    else:
        v = jnp.zeros(n, dtype=dtype)
        alpha = 0.0

    if alpha > 0:
        v = v / alpha

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1.0
    rhobar = 1.0
    cbar = 1.0
    sbar = 0.0

    h = v
    hbar = jnp.zeros(n, dtype=dtype)

    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = 0.0
    thetatilde = 0.0
    zeta = 0.0
    d = 0.0

    normA2 = alpha * alpha
    maxrbar = 0.0
    minrbar = 1e100
    normA = normA2**0.5
    condA = 1.0
    normx = 0.0

    istop = 0
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    normr = beta
    normar = alpha * beta
    if normar == 0 or normb == 0:
        return x, istop, itn, normr, normar, normA, condA, normx

    while itn < maxiter:
        itn += 1

        u = -alpha * u + A.matvec(v)
        beta = _norm(u)
        if beta > 0:
            u = u / beta
            v = -beta * v + A.rmatvec(u)
            alpha = _norm(v)
            if alpha > 0:
                v = v / alpha

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        hbar = -(thetabar * rho / (rhoold * rhobarold)) * hbar + h
        x = x + (zeta / (rho * rhobar)) * hbar
        h = -(thetanew / rho) * h + v

        betaacute = chat * betadd
        betacheck = -shat * betadd

        betahat = c * betaacute
        betadd = -s * betaacute

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = (d + (betad - taud) ** 2 + betadd * betadd) ** 0.5

        normA2 = normA2 + beta * beta
        normA = normA2**0.5
        normA2 = normA2 + alpha * alpha

        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        normar = abs(zetabar)
        normx = _norm(x)

        test1 = normr / normb
        test2 = normar / (normA * normr) if (normA * normr) != 0 else float("inf")
        test3 = 1.0 / condA
        t1 = test1 / (1.0 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        if itn >= maxiter:
            istop = 7
        if 1.0 + test3 <= 1.0:
            istop = 6
        if 1.0 + test2 <= 1.0:
            istop = 5
        if 1.0 + t1 <= 1.0:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if istop > 0:
            break

    return x, istop, itn, normr, normar, normA, condA, normx


def _norm(x: Any) -> float:
    import jax.numpy as jnp

    return float(jnp.linalg.norm(x))


def _sym_ortho(a: float, b: float) -> tuple[float, float, float]:
    """Stable symmetric Givens rotation."""
    if b == 0:
        return _sign(a), 0.0, abs(a)
    if a == 0:
        return 0.0, _sign(b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = _sign(b) / (1.0 + tau * tau) ** 0.5
        c = s * tau
        r = b / s
        return c, s, r
    tau = b / a
    c = _sign(a) / (1.0 + tau * tau) ** 0.5
    s = c * tau
    r = a / c
    return c, s, r


def _sign(value: float) -> float:
    return 1.0 if value >= 0 else -1.0
