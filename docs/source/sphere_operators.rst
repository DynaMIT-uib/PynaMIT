Sphere Operators
================

The ``pynamit.sphere`` package contains spherical representations,
surface bases, transforms, and solid-harmonic radial operations.

``Grid`` and the basis classes are spherical representations.  ``Grid``
stores sampled values, while ``SHBasis`` and ``CSBasis`` reconstruct
functions and provide surface operators.  Both basis classes implement
the same ``SurfaceOperators`` interface.

``SphericalTransform`` also belongs to this package.  It performs
analysis and synthesis between a surface basis and a spherical grid for
both scalar and tangential Helmholtz fields.

Horizontal Surface Convention
-----------------------------

Tangential Helmholtz coefficients are ordered as two scalar potentials:

``(curl-free potential, divergence-free potential)``.

The synthesized tangential field is

``F = -grad(phi) + rhat x grad(psi)``.

With that convention,

``surface_divergence(F) = -surface_laplacian(phi)``

and

``radial_curl(F) = surface_laplacian(psi)``.

These identities are exposed by the shared ``SurfaceOperators`` methods,
so code outside the basis implementations can ask for scalar evaluation,
surface gradients, Helmholtz synthesis, surface divergence, radial curl,
and surface Laplacian through the same interface for SH and CS bases.
The bases own these matrix/operator objects directly; transforms are
consumers of this interface rather than a separate evaluation layer.

Surface Bases Versus Solid Harmonics
------------------------------------

The surface basis describes functions on one spherical surface.
``SolidHarmonics`` is a separate object wrapping an ``SHBasis``.  It
describes how the wrapped angular coefficients participate in regular
and irregular three-dimensional Laplace solutions.  Radial operations
are deliberately not methods on ``SHBasis`` because a surface basis does
not, by itself, select a radial solution or physical coefficient
convention.

PynaMIT uses the geomagnetic reference-radius convention

``V_regular(r; R) = R sum(q_nm(R) (r/R)^n Y_nm)``

and

``V_irregular(r; R) = R sum(g_nm(R) (R/r)^(n+1) Y_nm)``.

Changing the reference radius from ``start`` to ``end`` therefore
scales regular coefficients by ``(start/end)^(1-n)`` and irregular
coefficients by ``(start/end)^(n+2)``.  The extra power comes from the
leading reference-radius factor ``R``, not from the normalization of
the angular spherical harmonics.

The stored PynaMIT poloidal coefficient ``m_nm`` is not either raw
potential coefficient.  At the reference sphere,

``q_nm = -(n+1) m_nm``

and

``g_nm = n m_nm``.

Consequently, ``B_r = n(n+1) m_nm Y_nm`` and
``(V_irregular - V_regular) / R = (2n+1) m_nm Y_nm``.  These conversions
are explicit ``SolidHarmonics`` operations.  No additional conversion
factor is needed for a reference-radius shift because the
degree-dependent coefficient conversions cancel in the ratio.

When the horizontal basis is CS, radial terms are handled by projecting
between CS horizontal coefficients and the SH basis wrapped by
``SolidHarmonics``.  This keeps the CS finite-difference operators local
to the surface while retaining SH angular coefficients for radial
Laplace physics.

Regularization
--------------

Current least-squares regularization is a degree-weighted spectral
penalty and therefore requires harmonic degree metadata.  The transform
constructs those penalties using the Helmholtz selector operators, but
the weighting policy itself is not part of the shared surface-basis
interface.
