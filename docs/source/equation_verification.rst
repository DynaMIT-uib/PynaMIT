Equation audit
==============

This audit records the maintained PynaMIT and Kompe mathematical implementation
reviewed in September 2026. It separates algebraic identities, numerical
approximations, physical assumptions, and external-provider contracts. It is
not a proof of every execution path or a certification of arbitrary simulation
resolution, parameters, or input data.

The equation-bearing implementations are grouped in the coverage ledger below.
The review combined independent vector-calculus derivations, analytic solutions,
numerical quadrature, existing regression tests, and comparison with published
equations and reference code. Agreement between two functions sharing an
implementation is not independent verification. In particular, the added
``test_physical_identities.py`` and ``test_secs_physical_identities.py`` derive
their expectations outside the operators under test.

Third-party model internals, historical scripts under ``scripts/simulation/legacy``,
and every individual empirical coefficient were not independently rederived.
Native-provider tests check the maintained adapters, not the empirical validity
of AMPS, Hardy/EUV, HWM, IGRF, or Apex models. Full production MAGE datasets were
not rerun as part of this audit.

Conventions that determine the signs
------------------------------------

* ``(r, theta, phi)`` is right-handed: theta is colatitude, increasing southward;
  phi increases eastward. Coordinate inputs are generally degrees, but angular
  derivatives are per radian. Cubed-sphere ``xi, eta`` are radians.
* Tangential physical arrays use ``(theta, phi)``. Provider ENU components map
  to spherical components as ``(r, theta, phi) = (up, -north, east)``.
  IAGA XYZ is north, east, down; these are not Earth-centred Cartesian axes.
* Core quantities use metres, seconds, tesla, V/m, A/m, A/m², siemens, and ohms.
  ``JS`` is a height-integrated sheet current, whereas ``jr`` is a volume
  current density at the upper ionospheric boundary, positive outward.
* Hall conductance/resistance are positive magnitudes. Hemisphere dependence
  enters through the signed magnetic unit vector, not by changing Hall's sign
  a second time. Inclination is measured from the horizontal, positive down:
  ``sin(I) = -Br / |B|``.
* Write ``grad_1`` and ``Delta_1`` for unit-sphere operators. The scalar
  evaluation derivative ``"phi"`` means ``(1/sin(theta)) partial_phi``, not
  the bare longitude partial derivative. At radius R, physical gradient and
  Laplacian carry factors ``1/R`` and ``1/R²`` respectively.

Independent induction and closure derivation
--------------------------------------------

The Helmholtz synthesis convention is

.. math::

   \mathbf F=-\nabla_1\Phi+\hat{\mathbf r}\times\nabla_1 W,
   \qquad \nabla_1\cdot\mathbf F=-\Delta_1\Phi,
   \qquad (\nabla_1\times\mathbf F)_r=\Delta_1 W.

PynaMIT stores radius-normalized electric potentials: stored Phi and W have
units V/m, and the physical potentials returned by result evaluation are R
times their scalar synthesis, in volts. An electric stream function is not
an additional electrostatic potential.

For one degree-n harmonic Y, let ``D = n(n+1)`` and let k denote PynaMIT's
private induced-poloidal coefficient (tesla). Differentiating scalar magnetic
potentials, rather than reading operator factors back from the code, gives

.. math::

   V^-=-R(n+1)k(r/R)^nY,\qquad
   V^+=Rn k(R/r)^{n+1}Y,\qquad \mathbf B=-\nabla V,

.. math::

   B_r(R)=DkY,\qquad
   \mu_0\mathbf J_S=\hat{\mathbf r}\times(\mathbf B^+-\mathbf B^-)
       =-(2n+1)k\,\hat{\mathbf r}\times\nabla_1Y.

The toroidal coefficient psi instead satisfies

.. math::

   \mathbf B_t=\hat{\mathbf r}\times\nabla_1\psi,\qquad
   \mathbf J_{S,\mathrm{cf}}=-\mu_0^{-1}\nabla_1\psi,\qquad
   j_r=(\mu_0 R)^{-1}\Delta_1\psi.

Thus current continuity is ``jr = -div_S(JS)`` and Faraday's law is

.. math::

   \dot B_r=-R^{-1}\Delta_1 W=R^{-1}DW,\qquad \dot k=W/R.

These factors distinguish physical Br from the private potential. Neither
the radius nor ``n(n+1)`` may be dropped when changing state coordinates.

For the thin-sheet Ohm law, write b for the magnetic unit vector and b_h for
its horizontal components. In the infinite parallel-conductivity limit with
sheet radial current zero, the horizontal resistance is

.. math::

   \eta_P={\Sigma_P\over\Sigma_P^2+\Sigma_H^2},\qquad
   \eta_H={\Sigma_H\over\Sigma_P^2+\Sigma_H^2},\qquad
   R_h=\eta_P(I_h-b_hb_h^T)+
       \eta_H\begin{pmatrix}0&b_r\\-b_r&0\end{pmatrix}.

For one height-independent horizontal wind,
``E_h = R_h JS - (u x B)_h``. The motional contribution in theta/phi order is
``(-Br*u_phi, Br*u_theta)``. For independently Pedersen- and Hall-weighted
winds, start with the three-dimensional wind-current moment

.. math::

   \mathbf q=\Sigma_P(\mathbf u_P\times\mathbf B)
      +\Sigma_H\,\mathbf b\times(\mathbf u_H\times\mathbf B),
   \qquad
   \mathbf E_{\mathrm{wind},h}
      =[-\eta_P\mathbf q_\perp-\eta_H(\mathbf q\times\mathbf b)]_h.

The independent test inverts the full 3-D conductivity tensor before taking
large parallel conductivity; it includes both signs of Br, a dip-equatorial
direction, and several height layers. It checks that equal weighted winds
reduce to the ordinary motional term. ``Q_eff`` uses PynaMIT's convention
``R_h Q_eff = E_wind,h``; it is not the same signed quantity as an arbitrary
author's effective wind current.

An especially useful whole-model check has uniform conductance and
``b = s*rhat``, where ``s = +/-1``. Every harmonic must obey

.. math::

   \dot B_{r,n}=-a_n B_{r,n}-s\eta_H j_{r,n},\qquad
   a_n={\eta_P(2n+1)\over\mu_0R}>0,

.. math::

   B_{r,n}(t)=B_{r,n}^{\mathrm{eq}}
      +(B_{r,n}(0)-B_{r,n}^{\mathrm{eq}})e^{-a_nt},\qquad
   B_{r,n}^{\mathrm{eq}}=-s\eta_H j_{r,n}/a_n.

The added test exercises the complete response, equilibrium, Faraday map,
and exponential integrator in both orientations. Pedersen always damps this
case; reversing the main field reverses Hall forcing, not dissipation.

Coverage ledger
---------------

Paths below are relative to the named repository. A test file is evidence for
the stated contracts, not a claim of exhaustive coverage of its module.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Equation family / implementation
     - Checks and qualifications
     - Test evidence
   * - Kompe: spherical coordinates and cell geometry
     - Cartesian/spherical conversions, ENU bases, proper rotations,
       ``2 atan2(|a dot (b x c)|, 1+a.b+b.c+c.a)`` solid angles.
       Global polygon areas sum to ``4*pi``. Longitude is a coordinate
       convention at a pole, not a unique physical direction.
     - ``test_spherical_coordinates.py``, ``test_mesh_interface.py``,
       ``test_global_cs_projection.py``
   * - Kompe: SH evaluation and normalization
     - Real sine/cosine harmonics, Schmidt factors, Legendre recurrences and
       differentiated recurrences, polar limits, degree/order indexing.
       ``Delta_R Y = -n(n+1)Y/R²``. Surface RMS normalization includes
       ``1/(4*pi)``; raw L2 does not.
     - ``test_spherical_transform.py`` (including independent Gauss-Legendre
       quadrature), ``test_basis_interface.py``
   * - Kompe: Helmholtz synthesis, gauges, smoothness
     - Curl-free sign, 90-degree rotation, div/curl blocks, area-mean gauge,
       constant nullspace and subset restrictions. Scalar penalty is gradient
       RMS; vector penalty is div-curl RMS, not the covariant gradient norm.
     - ``test_basis_interface.py``, ``test_coefficients.py``,
       ``test_spherical_transform.py``
   * - Kompe: solid-harmonic continuation
     - With reference-prefactor R in V, changing reference radius a to b
       scales regular coefficients by ``(a/b)^(1-n)`` and irregular ones by
       ``(a/b)^(n+2)``. Physical interior Br scales as ``r^(n-1)``.
     - ``test_basis_interface.py``;
       PynaMIT ``test_physical_identities.py``
   * - Kompe: cubed-sphere charts and vector components
     - Embedded Cartesian Jacobians, inverse/dual transformations, metric,
       determinant, rotations between faces, and regional chart orientation.
       Regional projection is a rotated global north-face chart.
       Global array order is face/xi/eta; regional order is eta/xi.
     - ``test_global_cs_projection.py``, ``test_regional_cubed_sphere.py``
   * - Kompe: CS finite differences and interpolation
     - Polynomial difference weights, cross-face barycentric interpolation,
       one-sided regional boundary stencils, coordinate-to-physical gradients,
       metric divergence, scalar and contravariant-vector remapping.
       These are approximations, not exactly mimetic surface calculus.
     - ``test_finite_differences.py``, ``test_basis_interface.py``,
       ``test_regional_cubed_sphere.py``, ``test_spherical_transform.py``
   * - Kompe: spherical SECS scalar/current/magnetic kernels
     - Green potential ``-log(sin(theta/2))/(2*pi)``, its negative gradient,
       ``cot(theta/2)/(4*pi*R)`` current, below/above magnetic branches,
       normalizations, induction image radius and amplitude.
       Magnetic fields checked against independent Biot-Savart surface
       integration and the Ampere sheet jump, not only SECSy formulas.
     - ``test_secs_basis.py``, ``test_secs_physical_identities.py``
   * - Kompe: Cartesian elementary currents
     - Radial and azimuthal currents, ring-integrated magnetic field, sheet
       jump, axial limits, and one-sided curl-free field. Positive Cartesian
       DF circulation is clockwise; positive spherical DF circulation is
       eastward about its pole. These established signs are not interchangeable.
     - ``test_cartesian_secs.py``
   * - Kompe: inclined current wedges
     - Semi-infinite Biot-Savart integral, projection onto the current line,
       radial-leg subtraction, ENU conversion. Positive current enters along
       the inclined outward ray and leaves radially. Direction reversal does
       not reverse this amplitude convention. Actual current rays remain
       singular; the backward extension is not a current ray.
     - ``test_secs_physical_identities.py`` (independent line quadrature),
       ``test_secs_basis.py``
   * - Kompe: LinearMap and tensor contractions
     - Composition, shaped application, conjugate adjoints, diagonal and
       pointwise maps, constrained coordinates, einsum contractions and
       ``diag(A* A)``. Materialization changes execution, not the operator.
     - ``test_linear_map.py``, ``test_least_squares_solver.py``
   * - Kompe: least squares and pseudoinverses
     - Weighted objective, square-root weights, Tikhonov scaling, nullspace
       elimination, normal systems, direct factors, spectral cutoffs,
       preconditioning, LSMR recurrences and CGLS. Algorithms do not give
       identical truncation at an identical numeric tolerance; see
       :doc:`numerical_model`.
     - ``test_pseudoinverse.py``, ``test_least_squares_solver.py``,
       ``test_spherical_transform.py``
   * - Kompe: affine exponential and sampled evolution
     - Augmented exponential ``exp(h*[[A,b],[0,0]])``; diagonal increment
       ``h*phi1(h*A)*b``, with ``phi1(0)=1``. No inverse of A is required.
       Euler interpolation does not change accepted steps. SciPy methods
       adapt over each fixed-forcing interval.
     - ``test_exponential.py``, ``test_linear_evolution.py``
   * - PynaMIT: MainField and magnetic coordinates
     - Dipole Br/Btheta, radial ``r^-2`` model, IGRF nT/km boundaries,
       GEO/MAG/SM rotations, modified-apex dual bases, conjugacy,
       ``r/sin²(theta)`` dipole shell and current mapping along B.
       Exact dipole-apex branch ambiguity now raises explicitly.
     - ``test_main_field.py``, ``test_kaiju_geopack_reference.py``
       (stored independent Kaiju Fortran rotations)
   * - PynaMIT: conductance coordinates and thin-sheet Ohm law
     - Reciprocal Pedersen/Hall conversion, dimensionless log magnitude and
       log ratio, stable reconstruction, signed resistance tensor, motional
       E, weighted-wind moments, ``Q_eff`` inversion and its regularization.
     - ``test_ionospheric_closure.py``, ``test_physical_identities.py``,
       ``mage/test_wind_forcing.py``
   * - PynaMIT: Joule heating
     - ``etaP * JS.T @ (I_h-b_h b_h.T) @ JS >= 0`` for the single-wind sheet;
       the antisymmetric Hall tensor contributes zero. This is frictional
       sheet-model heating, not generally ``JS dot E`` or the full
       height-integrated heating of vertically varying winds.
     - ``test_ionospheric_closure.py``, ``test_physical_identities.py``
   * - PynaMIT: ionospheric magnetic jump and boundary images
     - Poloidal/toroidal current relations derived above, regular/irregular
       continuation, reflected boundary factors and
       ``1/(1-(RI/RM)^(2n+1))``. Optional induced shielding and the prescribed
       outer-boundary field are distinct conditions.
     - ``test_magnetic_boundary.py``, ``test_magnetic_variables.py``,
       ``test_physical_identities.py``
   * - PynaMIT: poloidal field of inclined FACs
     - ``j(r)=jr(RI)*B(r)/Br(RI)`` along mapped footpoints, horizontal shell
       current ``j_h dr``, radial midpoint integration and image continuation.
       An independently constructed divergence-free twisted radial field
       gives an analytic degree-two integral, with and without an outer image.
     - ``test_magnetic_boundary.py``, ``test_physical_identities.py``
   * - PynaMIT: electrodynamic response and hemispheric coupling
     - Ohm maps compose explicit JS terms; toroidal coefficients solve radial
       current and conjugate-E residuals. Feedback elimination and forcing
       enter with the signs fixed by those equations. The conjugate condition
       is weighted least squares, not an exact boundary constraint.
     - ``test_electrodynamic_response.py``, ``test_physical_identities.py``,
       ``regression/``
   * - PynaMIT: induction and equilibrium
     - Physical-Br/private-k conversion, Faraday factor ``1/RI``, feedback,
       forcing, and minimum-norm equilibrium. Absolute integration tolerance
       is tesla and is transformed degree by degree. A singular equilibrium
       fit can retain a Faraday residual; it is not used to define exp(A).
     - ``test_evolution.py``, ``test_equilibrium_init.py``,
       ``test_physical_identities.py``
   * - PynaMIT: input times and MAGE matching
     - Seconds from a physical origin, MJD conversion, UTC, Kaiju whole-second
       rotation time, source-time matching, right-continuous held inputs.
       Each change splits integration at its timestamp, not at an output
       sample. Br is continuous; algebraic output at the change uses new input.
     - ``test_evolution.py``, ``test_field_time_series.py``,
       ``inputs/test_time_dependent_inputs.py``, ``mage/test_preparation.py``
   * - PynaMIT: MAGE preparation and projection
     - GAMERA background-field subtraction and reference-radius cubed scaling,
       trilinear volume centroids, true boundary solid angles, periodic and
       skewed-cell interpolation; ReMIX hemisphere/FAC conventions; TIEGCM
       layer integration, conductivity-weighted winds, and unit conversions.
       Floors and lower-dynamo extension are physical modelling choices.
     - ``mage/test_preparation.py``, ``mage/test_projection.py``,
       ``mage/test_wind_forcing.py``
   * - PynaMIT: empirical input adapters
     - Explicit Hall/Pedersen order, µA/m² to A/m², north/east to theta/phi,
       provider coordinate views, UTC, and quadrature addition of auroral,
       EUV, and starlight conductances. Empirical laws themselves are upstream.
     - ``inputs/test_external_input_coordinates.py``,
       ``inputs/test_native_fallback_pipeline.py``
   * - PynaMIT: output and observational comparison
     - Physical potential factors, component rotations, ground-field
       continuation, sheet current composition, time derivatives in seconds,
       interpolation on origin-subtracted timestamps, area-weighted fit errors.
       Observational missing-data and peak-selection rules are analysis
       conventions, not governing equations.
     - ``test_ground_field.py``, ``test_physical_identities.py``,
       ``visualization/test_time_series.py``, ``mage/test_projection.py``

Established corrections
-----------------------

The audit changed the following mathematical behaviour, without altering the
ionospheric governing equations or selecting new numerical solvers:

* Cartesian DF elementary-current magnetic fields used a normalized direction
  multiplied by a vanishing magnitude, producing NaNs on their nonsingular
  vertical axis. Rationalizing the expression gives, with
  ``d = hypot(rho, z)``, horizontal factor
  ``-C*sign(z)/(d*(d+abs(z)))`` multiplying ``(x,y)``, and ``Bz=-C/d``.
  This also preserves the small off-axis field without cancellation.
* The Cartesian curl-free magnetic field used ``isclose(z, 0)`` to select the
  above-sheet branch. Points strictly below the sheet could therefore acquire
  a field that must be zero. The branch now uses the exact sign of z.
* The inclined-current line integral had a removable singularity on the
  backward extension of a ray. If a is distance along the outward ray and
  d is distance to its endpoint, the integral is
  ``(1+a/d)/rho² = 1/(d*(d-a))`` for ``a<0``. Using the latter expression
  preserves both its axial limit and small off-axis values.
* Mapping an exact dipole apex to a different radius previously used a
  zero hemisphere sign and could return a point on the wrong field line.
  It now rejects that ambiguous request. Identity mapping remains valid.

No epsilon displacements were added to physical coordinates. Genuine SECS
point/ray singularities remain singular. The SECS field convention on a
zero-thickness sheet is a chosen one-sided limit or symmetric average, not
a uniquely defined value through a finite current layer.

The inclination definition, weighted-wind Hall cross product, Joule-heating
interpretation, SH regularizer norm, global mesh axis order, regional nominal
dimensions/area quadrature, and interhemispheric residual weight units were
also clarified in the implementation documentation.

Open numerical and physical qualifications
------------------------------------------

Global cubed-sphere conservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The global CS Laplacian is a collocated finite-difference approximation. It
annihilates constants, but its left nullspace does not coincide exactly with
the exact-cell-area weights. Thus ``area_weights @ Laplacian`` is not zero
at finite resolution. This can leave a nonzero area-integrated radial current
when current is formed from that discrete Laplacian. It is not cured merely
by fixing the potential's constant gauge.

For ``f=P4(cos(theta))`` on a unit sphere, ``Delta f=-20f`` and its continuum
integral is zero. The audit measured the following with the default stencil;
means and RMS values use normalized cell-area weights:

.. list-table::
   :header-rows: 1

   * - Cells per face edge
     - RMS(Delta_h f + 20 f)
     - Mean(Delta_h f)
   * - 8
     - 1.50989
     - 0.237936
   * - 16
     - 0.415729
     - 0.0658800
   * - 32
     - 0.109561
     - 0.0168008

A new regression checks convergence of both defects, without subtracting a
mean from the answer. The API's Helmholtz div/curl block identities use the
same discrete Laplacian; passing those identities is not proof of exact
discrete integration by parts or conservation. A conservative/mimetic
discretization would be a substantive follow-up numerical-method change,
requiring seam, nullspace, energy, spectral, and SH/CS convergence checks.
Silently removing current means would instead break the stored current's
exact relationship to its private potential.

Model assumptions and residuals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Infinite parallel conductivity and a thin current sheet are approximations.
  Horizontal E from the resistive formulation can be finite at the dip
  equator, but recovering radial E from ``E dot B=0`` or currents through
  ``1/Br`` is not generally defined there. This is not a complete electrojet
  or three-dimensional ionosphere model.
* A single conductivity-weighted wind is insufficient for all height-varying
  dynamos. Separate Pedersen/Hall first moments suffice for the implemented
  electric forcing, but not for heating, which also involves wind second
  moments. The saved sheet-model heating must not be described as that full
  height integral.
* Modified-apex base vectors are dual vectors with physical scale factors,
  not generally unit vectors. The IGRF/Apex adapter deliberately uses the
  documented spherical-geographic approximation rather than an ellipsoidal
  geodetic conversion. Do not infer exact IGRF flux mapping from dipole tests.
* Interhemispheric E matching is an assumed coupling model implemented as a
  weighted residual. The weight has units S/m and is squared in the objective.
  Check current/E residuals as well as the linear solver's stopping condition.
* Regional mesh ``length`` and ``width`` retain the SECSy convention
  ``angular_span=atan(dimension/radius)``. They approximate surface distances
  for small patches. Regional cell areas use midpoint metric quadrature;
  global CS cell areas use exact spherical polygons.
* Input projection truncation, angular resolution, radial FAC quadrature and
  its upper cutoff, conductance floors, and the lower-dynamo extension require
  case-specific sensitivity checks. Positivity of reconstructed conductance
  does not prove that its source-space floor is preserved after projection.
* Euler has first-order global accuracy and a stability restriction determined
  by the generator; its one-step truncation error is second order. Adaptive
  solvers and exponential propagation do not remove spatial discretization
  error. Float64 tests do not establish equivalent high-accuracy behaviour
  with JAX x64 disabled.

Comparison with references
--------------------------

The main reference was `Laundal et al. (2025), Global inductive
magnetosphere-ionosphere-thermosphere coupling
<https://doi.org/10.5194/angeo-43-803-2025>`_. Its main equations and appendices
were checked against the implementation and the independent identities above.
The typeset PDF matters here: text extraction can lose cross-product order,
superscripts, and minus signs.

Several printed signs should not be copied literally:

* The sentence before Eq. (20) uses a positive potential gradient, inconsistent
  with the surrounding magnetic-potential equations.
* The Hall wind-current cross product in Eq. (A4) is reversed relative to
  Eq. (A1). The conductivity law requires ``b x (u_H x B)``.
* Appendix B's current-potential and Hall-forcing signs are inconsistent with
  the declared downward field and with Eq. (39). For downward b, writing
  ``q=-Br/n`` gives
  ``q'=-a*q + etaH*(n+1)*psi/(mu0*R)``, agreeing with Eq. (39).

These are consistency findings from this audit, not an author-issued erratum.
The implementation was not changed to reproduce the conflicting signs.

`Laundal et al. (2022), Local Mapping of Polar Ionospheric Electrodynamics
<https://doi.org/10.1029/2022JA030356>`_ and the
`SECSy implementation <https://github.com/klaundal/secsy>`_ provide independent
convention and kernel comparisons. The Lompe preprint's Eqs. (5)-(13) agree
with the spherical current and magnetic-field branches. Its electrostatic,
approximately radial-field assumptions must not be imposed on inductive
PynaMIT. SECSy comparisons were supplemented by Biot-Savart quadrature rather
than treating identical reference expressions as proof.

Reproducing the checks
----------------------

Run PynaMIT's fallback suite on both backends, followed by focused native
adapter equivalence::

    pytest -q --backend numpy --backend jax --data-source fallback
    pytest -q -m native_input_validation --backend numpy --backend jax --data-source native

In the matching Kompe source checkout, run both backend configurations::

    KOMPE_USE_JAX=0 pytest -q
    KOMPE_USE_JAX=1 JAX_ENABLE_X64=1 pytest -q

The targeted new checks can be run with PynaMIT's
``tests/test_physical_identities.py`` and ``tests/test_main_field.py``, and
Kompe's ``tests/test_secs_physical_identities.py``,
``tests/test_cartesian_secs.py``, and the degree-four convergence check in
``tests/test_basis_interface.py``. Keep the two source installations aligned;
running new tests against an older installed Kompe does not audit this code.
