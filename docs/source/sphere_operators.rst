Sphere Operators
================

The ``pynamit.sphere`` package contains the horizontal surface bases and
operators used by the rest of PynaMIT.  Both spherical-harmonic and
cubed-sphere bases implement the same surface-operator interface.

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

Horizontal Basis Versus Radial Continuation
-------------------------------------------

The horizontal basis describes functions on the ionospheric surface.  A
radial Laplace continuation is a separate capability: it describes how
SH surface coefficients continue into regular or irregular 3D Laplace
solutions above and below the sheet.

When the horizontal basis is CS, radial-continuation terms are handled
by projecting between the CS horizontal coefficients and the SH
radial-continuation basis.  This keeps the CS finite-difference surface
operators local to the surface while retaining the SH representation for
3D Laplace-continuation physics.

Regularization
--------------

Current least-squares regularization is a degree-weighted spectral
penalty and therefore requires harmonic degree metadata.  The evaluator
constructs those penalties using the Helmholtz selector operators, but
the weighting policy itself is not part of the shared surface-basis
interface.
