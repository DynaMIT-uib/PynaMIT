Physics and numerics
====================

This page records the mathematical contracts behind the ownership described
in :doc:`architecture`. See :doc:`sphere_operators` for spherical conventions
and :doc:`input_workflows` for provider and projection boundaries.

One shared fit policy
---------------------

``least_squares_solver`` and ``least_squares_tolerance`` configure sampled
input projections, wind-to-``Q_eff`` coefficient fits, and the instantaneous
toroidal-potential response. They are saved with prepared inputs and inherited
by ``Simulation.from_inputs`` and ``run_from_inputs`` unless explicitly
overridden. Input setters take physical data, weights, and regularization,
not separate solver/tolerance menus::

    preparation = InputPreparation(
        horizontal_basis_kind="CS",
        least_squares_solver="lsmr",
        least_squares_tolerance=1e-10,
    )
    # preparation.set_conductance(...), preparation.set_boundary_jr(...)
    simulation = Simulation.from_inputs(preparation)

Defaults remain SH ``normal_pinv``, CS ``lsmr``, and tolerance ``1e-15``.
``KOMPE_LEAST_SQUARES_SOLVER`` supplies the default algorithm when set.
There is no tolerance-driven or problem-inspection algorithm switching.
The common tolerance has the selected algorithm's conventional meaning:

* ``svd`` discards singular values below a relative cutoff.
* ``normal_pinv`` applies that cutoff to normal-matrix eigenvalues (squared
  singular values). This preserves its historical behavior; the same number
  does not imply the same spectral filtering as SVD.
* ``lsmr`` uses residual and normal-residual convergence tolerances.
* ``cgls`` uses relative convergence of the normal equations.
* ``normal_solve`` uses reusable direct factors, with no spectral truncation.
  It requires a nonsingular normal matrix after removing known gauges;
  tolerance does not affect it. Unregularized native CS analysis reuses its
  sparse constrained LU inverse, and full-rank SH analysis can reuse Cholesky.

Weights, regularization, and gauges define the mathematical problem and do
not select its algorithm. Helmholtz fit gauges fix constant potentials without
changing the field objective or the scale of its smoothness penalty. Fits
use independent zero-mean coordinates rather than adding gauge penalties.
A coefficient subset is constrained only if it still represents a constant.
Wind-to-``Q_eff`` regularization remains the absolute coefficient penalty
``lambda * ||Q_eff||²``. It is expressed with a structured diagonal operator,
so it works identically with every algorithm and with right preconditioning.

Fixed geometric inverses, surface Poisson maps, small pointwise coordinate
solves, and the compact equilibrium pseudoinverse are not configurable input
fits. Their reusable factorizations remain explicit at their mathematical
boundaries. In particular, evolution does not replace a fixed geometric map
with adaptive iterative analysis when the fit solver changes.

Numerical operators and value identity
----------------------------------------

``LinearMap`` is the common operator abstraction for dense, sparse,
matrix-free, and structured-einsum calculations. Least-squares problems and
physical compositions should retain ``LinearMap`` objects until an explicit
representation is genuinely required. ``to_matrix()`` returns a flat 2-D
matrix, while ``to_array()`` retains the shaped domain and codomain axes. The
tensor helper module contains only
contractions and pseudoinverses that still operate on multidimensional arrays;
do not add parallel wrappers for operations already expressed by
``LinearMap``.
``operator(values)`` applies the map's declared input/output shapes and retains
trailing batch axes. Physics code need not flatten fields or repeat the map's
output shape. ``@``, ``matvec``, and ``matmat`` retain flat linear-algebra semantics.
Creating an explicit map is also an execution decision. The response layer
prepares compact maps once, when their cached operator is first constructed;
large CS surface maps remain structured. Inspection and evolution use that same
operator, without a second set of runtime aliases or caches. Plotting and export
code should otherwise retain structured maps until explicit values are needed.
When a rectangular map must become explicit, materialization probes the
smaller side: input columns for tall maps and adjoint output rows for wide
maps. This is important for rectangular surface-to-poloidal and
boundary-current-to-gap-field maps, whose compact physical output should
determine temporary memory use.

Sparse equality-constrained least-squares factorization is a solver-layer
primitive rather than a cubed-sphere implementation detail. Native CS
Helmholtz inverse maps supply their sparse synthesis matrix and two constant
constraint rows to that primitive, then retain the resulting analysis as a
``LinearMap`` for both NumPy and JAX arrays. This avoids a dense
``(2 Ncs_surface)^2`` pseudoinverse while preserving the same weighted field
fit and fixing both potential gauges exactly. Request these fixed maps through
``SphericalTransform.helmholtz_analysis_operator``. ``analyze_helmholtz`` also
reuses this factorization when ``normal_solve`` is selected: this is the same
direct normal-equation solve, not an algorithm substitution. Iterative and
pseudoinverse selections retain their convergence and cutoff semantics.
Mean-free SH Helmholtz inverse maps use the corresponding dense full-rank
factorization. Their coefficient spaces omit both constant gauges, so a
cached normal-system factor replaces the tall SVD while retaining a
structured adjoint. Spaces retaining a mean mode, undersampled grids, and
otherwise rank-deficient transforms keep the pseudoinverse fallback.

``SphericalGrid`` is an immutable coordinate value. Its coordinate signature defines
grid-value compatibility and remapping identity. Its separate analysis
signature also includes optional area weights: equal coordinates can share
synthesis matrices while requiring different weighted least-squares analyses.
Transform caches must choose deliberately between those two identities.
Likewise, basis coefficient compatibility is distinct from its evaluation
signature. A requested SH Legendre algorithm is preserved by ``with_basis``
even when the coefficients themselves are interchangeable.
Mean-freedom is owned by coefficient spaces, not by post-step cleanup. The
schema's ``sh_basis`` retains the mean/monopole term for quantities such as
conductance, while ``mean_free_sh_basis`` is used for radial magnetic fields.
The evolving ``induced_Br`` at ``RI`` and prescribed ``boundary_Br`` at ``RM``
therefore live directly in the mean-free poloidal SH space.
``boundary_jr``, ``Phi``, and ``W`` live in the selected horizontal surface
space. In CS mode the surface potentials use an explicit zero-area-mean gauge
constraint. Saved ``boundary_jr`` is kept in the exact range of the discrete
surface Laplacian instead of applying a second, slightly different
area-mean projection. Each stored representation supplies its own synthesis
operator on the model grid, so ``SimulationGeometry`` does not carry a
parallel full-scalar transform.

Coefficient values are ordinary backend arrays. Ownership is established where
values become persistent state: ``FieldTimeSeries`` owns inserted rows,
``Simulation.set_state`` owns the live field, and ``ElectrodynamicResponse``
owns its fixed conductance snapshot. Numerical evaluation does not repeatedly
wrap or copy these arrays. ``CoefficientSpace`` supplies layout and gauge
normalization, without a second value-container representation.

Basis and grid metadata follow the same rule. Coefficient-index names are
tuples, and coefficient indices, CS coordinates, metric tensors, and cell
areas are owned read-only arrays. A basis cache key and its persisted
coefficient identity therefore cannot diverge through mutation by a caller.

Spatial bases
-------------

Spherical harmonics and cubed-sphere support have different implementation
needs, but they should present the same public basis contract wherever
possible. ``GlobalCSBasis`` is Kompe's public cubed-sphere basis. Its private
implementation is split into:

* ``cs_coordinates`` for panel and coordinate transforms;
* ``global_mesh`` for mesh shape and indexing;
* ``global_remapping`` for scattered scalar and vector remapping;
* ``global_differencing`` and ``finite_differences`` for derivative
  stencils and sparse operators; and
* ``cs_vectors`` for vector-basis conversions.

Prefer adding focused CS behavior to one of these collaborators instead of
growing ``GlobalCSBasis`` again. ``GlobalCSProjection`` owns the public
continuous coordinate and vector transformations; ``GlobalCSMesh`` owns
sampled geometry; and ``GlobalCSBasis`` owns expansion, interpolation, and
closed-surface differential behavior. Keep the ``CS`` abbreviation in class
names that represent actual objects.

``kompe`` is a warranted standalone package because its bases, grids,
analysis/synthesis transforms, and solid-harmonic continuation form a coherent
numerical domain that can be used without a PynaMIT simulation.

``geomagnetism`` is the corresponding standalone physical domain. ``MainField``
owns background-field models, field-line and conjugate mapping, magnetic
coordinates, and apex basis vectors. ``MainField.evaluate(grid, radius)``
returns ``(Br, Btheta, Bphi)`` directly on the active array backend. Its
bounded cache reuses values for equal coordinate signatures, radii, and
backends. ``unit_vector``, ``horizontal_to_apex_array``, and
``radial_to_apex_scale`` expose the related local geometry without another
bound evaluation object. The ``(2, 2, N)`` array contains one pointwise
component transform per sample; sampled fields use component or value names
instead.

The simulation frame is explicit and follows the main-field kind.
``kaiju_dipole``, ``igrf``, and ``radial`` use
``geocentric_geographic`` coordinates, so the SH and cubed-sphere positions
denote locations on Earth directly. The generic idealized ``dipole`` model
uses ``centered_dipole`` coordinates. For ``kaiju_dipole``,
``MainField`` alone owns the fixed GEO-to-MAG rotation: it evaluates the
analytic dipole, field-line mapping, conjugacy, magnetic latitude, and apex
basis vectors in MAG and returns coordinates and vector components in GEO.
The Kaiju/Geopack dipole coefficients and axis are frozen at
``main_field_epoch``. ``SM`` remains a timestamped external-source coordinate
system whose Sun-facing longitude origin is transformed at every source time;
it is never the coordinate space of the model coefficients. The background
field is not re-tilted every timestep.

Do not promote every focused module to a package. A standalone package is
warranted when the concept has a reusable public vocabulary and a dependency
boundary of its own. The PynaMIT boundary, ionospheric closure, and induction
equations are mutually coupled parts of one simulation model, so they remain
together under ``simulation.electrodynamics``.

Electrodynamic physics
----------------------

The ``electrodynamics`` modules follow the direction of the model equations
instead of mirroring individual coefficient names:

* ``electrodynamics.magnetic_boundary`` maps the physical
  ``induced_Br``, ``boundary_Br``, and ``boundary_jr`` quantities to the
  derived horizontal sheet current ``JS``. Solid harmonics own generic radial
  continuation; this module owns the particular potential jump and shielding
  relations used by PynaMIT, including radial shell integration of the gap
  field created by field-aligned current. That integral composes field-line
  mapping, current conversion, and solid-harmonic response in one function;
  ``SimulationGeometry`` owns reuse and cache identity of the resulting matrix.
  The private poloidal and toroidal potentials
  appear here only as convenient operator coordinates.
* ``electrodynamics.ionospheric_closure`` applies the height-integrated
  Ohm-law closure. It maps physical Pedersen/Hall conductance or resistance
  into the canonical log-conductance coordinates, reconstructs the resistance
  tensor, and maps neutral motion or sheet current through the magnetic
  geometry to ``E``. Results evaluation uses the same direct log-to-resistance
  conversion, preserving resistance values when intermediate conductance
  components would underflow. It owns both the direct grid law
  ``E = R JS - u x B`` and the coefficient-operator compositions used by
  ``ElectrodynamicResponse``. It also owns the Pedersen/Hall geometry tensors
  and collisional Joule-heating kernel. Pointwise vectors and tensors keep
  their component axes first, followed by grid or time axes; these kernels
  do not require callers to flatten sampled fields. Joule heating is the Pedersen
  dissipation ``etaP * J.T @ P @ J``; ``J dot E`` is electromagnetic work and
  is not generally the same quantity when neutral motion contributes to the
  closure. Its functions remain numerical kernels; iteration over input times
  and coefficient storage belong to ``_InputProjector``.
* ``electrodynamics.induction`` owns Faraday's law for physical
  ``induced_Br`` and its instantaneous equilibrium. Generic Euler, exponential,
  and adaptive integration belong to Kompe's linear-evolution primitive. It integrates in the private
  induced-poloidal-potential coordinate where that improves conditioning, then
  converts exactly at the module boundary. Motion of an SM forcing pattern
  appears through timestamped SM-to-GEO preparation, not through a rotating
  model basis.
  The SciPy path completes its forcing calculation on the selected array
  backend, then transfers the forcing and initial state at the solver
  boundary. Intermediate coordinate conversions do not round-trip through
  the CPU.

``SimulationGeometry`` supplies simulation-specific grids, magnetic-field factors,
transforms, radial field-line mapping, exact physical-to-private coordinate
maps, and interhemispheric constraint geometry to these equations. It is a
numerical object: persisted xarray values are unwrapped at construction, and
``SimulationResults`` alone wraps the numerical ``gap_Br_response`` for storage.
``ElectrodynamicResponse`` receives that geometry, owns the fixed conductance
at one simulation time, solves the toroidal-potential constraint behind
``boundary_jr``, and caches the composed operators whose values depend on the
current resistance distribution. It exposes named physical operators for
inspection; explicit arrays and matrices are obtained from those same maps.
Keeping those compositions with their
caches avoids split ownership between an operator facade and the response
object whose private state they mutate.

The combined field that drives the toroidal-potential response is named
``driving_E``. It can contain one neutral-wind representation, a
``boundary_Br`` response, or an ``induced_Br`` response. The name therefore describes
its role without suggesting that it contains only one neutral-wind
representation. The private toroidal potential then completes the response
required by the ``boundary_jr`` and optional interhemispheric constraints.
``E_neutral_wind`` is the public name for an externally prepared equivalent
neutral-wind electric field. It does not claim that the stored field is the
total electric field. In the MAGE workflow it is derived from separate
Pedersen- and Hall-weighted winds; the model composes the other terms
while solving the closure.

The poloidal and horizontal surface spaces coincide in the default SH mode.
They are intentionally distinct in CS mode. ``surface_to_poloidal_operator``
is the only bridge in Faraday's law: it projects the surface ``W`` potential
onto the configured poloidal harmonics.
``boundary_jr_to_gap_Br_operator`` has the corresponding rectangular shape.
Its input is radial current at ``RI+`` and its output is the unshielded
poloidal ``Br`` produced in the gap at ``RI``. That gap field and the inward
continuation of ``boundary_Br`` are both external-source radial fields, so the
same divergence-free ionospheric sheet-current response shields them. This
factorization prevents unobservable high-resolution CS modes from being
carried as part of ``induced_Br``.

Expensive optional geometry follows use rather than construction. The
``gap_Br_response`` is reused when persisted, but a new one is not built merely
to construct ``Simulation`` or project inputs. It is constructed and saved
when an equilibrium or dynamic path first requests model output. When PFAC
coupling is disabled (or the main field is radial), the optional contribution
is represented by absence rather than by constructing and multiplying a dense
zero matrix.

Surface-sized operators should remain structured in CS mode. In particular,
native Helmholtz analysis, wind and sheet-current closure compositions, and
ordinary runtime toroidal-potential responses must not materialize dense
surface maps.
The surface-to-poloidal bridge composes a factorized full-column-rank SH
analysis with the horizontal synthesis operator; native CS nodal synthesis
therefore stays an implicit identity rather than allocating a dense grid-sized
identity. The analogous poloidal-current fit used while constructing the gap
response also keeps a factorized normal system and an implicit adjoint instead
of forming an SVD pseudoinverse of its tall grid matrix.
The compact poloidal feedback matrix remains intentionally dense because its
matrix exponential and equilibrium pseudoinverse require an explicit
poloidal operator. The equilibrium response composes that compact
pseudoinverse with the structured surface-to-poloidal bridge; its full
cross-space matrix is materialized only when explicitly requested for
diagnostics. The generic dense ``normal_pinv`` solver remains the SH default
and is retained for reproducibility. CS toroidal-potential solves default to
matrix-free ``lsmr``; ``cgls`` and the ``jacobi`` preconditioner remain
explicit alternatives when their cost is justified for a particular system.

The toroidal-potential runtime follows the same rule. A single active
``boundary_jr`` or driving-E field is assembled as one physical least-squares
right-hand side and solved directly. The full
``boundary_jr_to_toroidal_potential_operator`` and
``driving_E_to_toroidal_potential_operator`` remain available for matrix
diagnostics, but normal simulation steps do not construct them.
Interhemispheric induction solves only the source columns reachable from
``induced_Br``, rather than first constructing the response to every possible
horizontal-E coefficient.

Magnetic-boundary maps also retain structured ``LinearMap`` compositions.
This matters in CS mode, where the native gradient and Helmholtz synthesis
operators are sparse. Repeated E-response maps are cached densely for compact
poloidal inputs and when the horizontal and poloidal SH spaces coincide; large
CS surface-to-surface maps remain structured until an explicit matrix is
requested.

The persisted input artifact is named ``conductance``. Its canonical scalar
fields are
``log_conductance_magnitude = log(hypot(SigmaP, SigmaH) / 1 S)`` and
``log_hall_to_pedersen_ratio = log(SigmaH / SigmaP)``. The fixed one-siemens
reference makes the first logarithm dimensionless without changing numeric
values already expressed in siemens. Both components are required to be
strictly positive. Fitting these two unconstrained coordinates guarantees
positive reconstructed Pedersen and Hall conductance. It also treats the
reciprocal resistance naturally: conductance and resistance have opposite
log magnitudes and the same Hall/Pedersen ratio.

``set_conductance`` is the canonical physical input API.
``set_coefficients('conductance', values)`` accepts already-projected
log-coordinate arrays in a dictionary keyed by the two canonical field names.
``set_resistance`` remains a
sample-level convenience for physical resistance values. It maps the
reciprocal resistance magnitude and unchanged Hall/Pedersen ratio directly
onto the same canonical log coordinates, without constructing intermediate
conductance components. A response owns two fixed coefficient arrays in its
shared conductance space, without per-field wrappers. It synthesizes them once,
reconstructs the resistance tensor on the model grid, and caches the
closure-dependent operators. ``simulation.response_at_time(t)`` returns a
response without changing simulation time. Equal conductance snapshots reuse
the previous response; changed values construct a new one, leaving previously
returned responses intact. Comparison and fingerprinting use the selected CPU
coefficients before transfer to the numerical backend.

``_TimeEvolution`` selects winds and boundary forcing independently, retaining
backend coefficient arrays and their noninductive solution until the forcing
or conductance changes. Selecting an earlier time or replacing inputs drops
forcing that is no longer present. The wind-to-electric-field map depends only
on geometry and lives on ``SimulationGeometry``.

New conductance creates new resistance-dependent operators. The toroidal
fit, its prepared solver, and its boundary-current response are retained when
there is no interhemispheric electric-field constraint: their operators depend
only on magnetic geometry. With that constraint enabled they are rebuilt;
``reuse_preconditioner`` may retain the preconditioner as configured. New
couplings should place their cached quantities with the physical inputs that
determine them in ``response.py``.

The shared surface convention is
``F = -grad(phi) + rhat x grad(psi)``. Stored ``Phi`` and ``W`` are the
curl-free and divergence-free electric-potential coefficients normalized by
``RI``; they therefore have units of V/m, while visualization multiplies them
by ``RI`` to display volts. Stored magnetic variables are instead physical:
``induced_Br`` is the continuous induced radial field at ``RI``,
``boundary_jr`` is radial current density at ``RI+``, and ``boundary_Br`` is
the prescribed radial field at ``RM``. Private potential coordinates obey
``boundary_jr = RI / mu0 * surface_laplacian(toroidal_potential)``,
``induced_Br = -RI**2 * poloidal_laplacian(induced_poloidal_potential)``, and
``d(induced_poloidal_potential)/dt = W / RI``. Exact forward and inverse maps
keep these numerical coordinates out of the public schema without changing
the established sign, radius, or time-evolution convention.

The CS toroidal-potential system has one physically irrelevant constant gauge.
The toroidal ``LeastSquaresProblem`` declares the surface-mean row through
``constraints=``. Kompe enforces zero-area-mean potentials and returns full
coefficient fields, without adding an artificial residual or requiring
PynaMIT to reconstruct independent coordinates.
This is a constraint, not Tikhonov regularization:
``toroidal_potential_regularization_lambda`` remains an optional numerical
policy damping observable directions within that constrained space. Its
relative scale is determined by the physical residuals alone, not gauge rows.

``SimulationGeometry`` names describe the physical map rather than the symbols
used in one derivation. In particular, ``pedersen_geometry_tensor``,
``hall_geometry_tensor``, and ``wind_motional_E_tensor`` are pointwise maps,
while ``induced_poloidal_potential_faraday_rate_scale`` converts the
divergence-free electric potential to the private potential rate.
``boundary_jr_to_gap_Br_matrix`` denotes the specific PFAC-derived physical
map from upper-boundary current to unshielded gap radial field; its persisted
artifact is ``gap_Br_response``.

The interhemispheric names distinguish a physical region from a boundary
condition. ``interhemispheric_coupling_latitude`` bounds the low-latitude
region where conjugate points are compared; it is not another magnetic
boundary. ``radial_current_constraint_operator`` is the assembled map used by
the toroidal-potential solve: it is the local radial-to-apex map outside that
region and the local-minus-conjugate map inside it.
``interhemispheric_electric_field_weight`` is specifically the relative
least-squares weight of the conjugate electric-field residual. Conjugate grids,
transforms, and masks are absent when this coupling is disabled or cannot
apply to the radial-field model.

Response and evolution
----------------------

``Simulation`` owns the live ``current_time`` and ``induced_Br`` and exposes
``evolve_to_time``. It delegates input scheduling, sampling, and writes to
``_TimeEvolution``. Faraday's law, physical tolerances and coordinate conversion
belong in ``electrodynamics.induction``. Kompe's
``prepare_linear_evolution(A, b, ...)`` owns reusable integration of
``x' = A x + b``, without PynaMIT-specific state or units.
Conductance-dependent operators belong in ``ElectrodynamicResponse``.

There are three separate clocks:

* Input timestamps mark changes in held forcing and conductance. Integrators
  stop exactly at these changes, even between requested outputs.
  Identical consecutive coefficient records are not changes and do not
  interrupt accepted solver steps.
* ``output_interval`` selects uniform output spacing in seconds, anchored at
  ``t0`` (default 0.1 s). Alternatively, ``output_times`` gives an increasing
  sequence of seconds after ``t0``. Explicit times do not add an initial row;
  ``output_times=[final_time]`` requests only the final state.
* Euler's ``dt`` is its fixed internal step (default 0.5 ms). SciPy chooses
  adaptive internal steps using ``rtol`` and ``atol``. Exponential evolution
  has no artificial internal timestep. Supplying ``dt`` to either of the
  latter integrator families is an error.

At a change timestamp the integrated magnetic state is continuous: the old
inputs govern the interval ending there, and the new inputs govern the next
interval. Algebraic output at that timestamp uses the new inputs. Input selection
matches equivalent floating-point clocks within four float64 rounding units,
not a fixed microsecond window. Distinct sub-microsecond requests remain distinct.
MAGE keeps its nominal TIEGCM clock for all projected input streams; the separate
GAMERA/ReMIX source offsets remain provenance and are not activation-time shifts.

The final time is always retained as a restart checkpoint. A checkpoint cannot
reconstruct missing earlier outputs; start from an earlier trajectory instead.
Euler uses its linear within-step interpolant at observation times, without
committing those observations as integration steps. Input changes and the final
time do end an accepted step, potentially fractionally. Thus output frequency
does not alter Euler's trajectory, but stopping and restarting at a fractional
step can differ from an uninterrupted run. Align restart times with Euler steps
when exact continuation equivalence matters.

SciPy runs one adaptive solver per constant-input interval and streams samples
from its dense interpolant. It does not restart at each sample or collect a full
trajectory before writing. The fixed rate matrix and completed forcing cross the
CPU boundary once per prepared response/forcing; the initial state crosses once
per interval. The integration tolerances are ``rtol=1e-3`` and
``atol=1e-12`` tesla by default, applied to physical induced-Br coefficients.
The diagonal conversion ``P[n,m] = Br[n,m] / (n*(n+1))`` converts the absolute
tolerance to the private solver coordinates, preserving the same error bounds.
This small tolerance vector also crosses the CPU boundary at preparation.
Integration tolerances are independent of the least-squares fit policy.

For constant inputs the induction equation is ``dP/dt = A P + b``.
Kompe's ``affine_exponential(A, b, duration)`` returns the propagator and
forcing increment. Dense propagation exponentiates an augmented matrix with
only one extra row and column. It requires neither an inverse nor an equilibrium
and retains growing forced null modes. Diagonal rates stay vectors, including
exactly zero rates. Dense exponentials materialize a matrix-free generator;
this is not an iterative exponential-action method. The prepared stepper caches
four recent duration maps for its fixed response and forcing. For uniform
outputs it reuses the requested interval as the key when timestamp subtraction
differs only by roundoff; input-boundary durations remain distinct.

``equilibrium_induced_Br`` remains a minimum-norm diagnostic and initialization
option. A singular or truncated fit can leave a residual; inspect
``induced_Br_time_derivative`` to assess the remaining evolution. No integrator
requires this estimate to be an exact equilibrium.

Kompe returns bounded blocks with shape ``state_shape + (n_samples,)``.
PynaMIT converts these blocks to physical magnetic coefficients and evaluates
output fields in batches. Euler and exponential array kernels execute sample
blocks inside JAX; SciPy keeps one adaptive solver and transfers a block of
samples at a time. Matrix-free Euler remains matrix-free.

Output fields remain backend arrays until ``samples_per_write`` triggers an
xarray update and optional disk persistence. Completed blocks become visible
without waiting for the complete trajectory; the final checkpoint is always
retained. Interruption flushes completed samples to memory and propagates the
exception. Batch boundaries do not restart internal stepping or move input
changes, and do not change which physical samples are retained.
Output cadence and write batching may change on continuation without changing
the trajectory identity. New regular sampling applies from the active checkpoint;
it does not fabricate missing historical output. Explicit requests for missing
earlier samples are rejected.

``equilibrium_coefficients(time)`` calculates independent equilibria without
changing state or histories. Scalar or 1-D time queries retain their order;
multiple times sharing conductance use a batched forcing solve. Conductance
must exist at every requested time, while optional sources are zero before
onset. Held inputs are the default, matching dynamic evolution. Explicit
``interpolation=True`` calculates diagnostics under interpolated inputs; it
does not change the live trajectory's held-input policy.

``sample_equilibria(times)`` records these diagnostics in bounded batches.
Earlier times may be added independently of the dynamic trajectory. No
integrator or initial condition is involved. ``evolve_to_time`` always evolves
the dynamic state; ``sample_equilibrium=True`` also records equilibria at its
output times. The high-level prepared-input workflow and GUI can still select
dynamic, equilibrium-only, or combined output, dispatching to these operations.
An equilibrium-only saved run has no dynamic state to resume.
The prepared-input workflow manifest is version 7 and
records output timing and integration tolerances. Older workflow manifests fail
the explicit version check; saved field data and the simulation-settings schema
are unchanged and remain accessible through ``Simulation.from_directory``
and ``SimulationResults.from_directory``.

To initialize an equilibrium explicitly, calculate its coefficients and pass
``induced_Br`` to ``set_state``. ``record_state()`` records the live dynamic
state without evolution; ``save=False`` suppresses the optional disk write.
Setting a state before later recorded dynamic output is rejected because
retaining both would create two trajectory branches in one history.

``ElectrodynamicResponse`` stores fixed conductance coefficients and the
algebraic response implied by that resistance. Winds, effective currents, and
magnetic-boundary forcing are explicit keyword arguments to
``solve_noninductive_response``. Its coefficient arrays may have trailing,
broadcast-compatible batch axes for independent cases with fixed conductance.
The evolving
``induced_Br`` is an explicit value owned by ``Simulation``. Equilibrium
estimates are separate diagnostic values, never aliases of the live field.
``_TimeEvolution`` passes those coefficient vectors into response methods.
The persisted artifact name
``dynamic`` identifies the time-dependent output stream; ``equilibrium``
identifies the instantaneous zero-Faraday-rate comparison.

Live state and restoration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Opening a saved simulation restores its latest dynamic checkpoint once.
Subsequent evolution uses the live field, even if recorded rows are edited
or removed. ``set_state(Br, time=...)`` installs an owned backend array without
recording output. ``restore_state(results, time=...)`` restores an exact dynamic
checkpoint; it does not interpolate, infer missing history, or delete later
records. Use a new simulation for an earlier branch::

    branch = Simulation.from_inputs(simulation.inputs)
    branch.restore_state(simulation.results, time=checkpoint_time)
    branch.evolve_to_time(final_time)

Restoration checks coefficient layout, radius, coordinate frame (including
dipole epoch), and time origin. ``Simulation.output_coefficients()`` evaluates
the live field and held inputs independently of recorded results. Saving
persists recorded histories; call ``record_state()``
first to record a newly installed or edited state. This distinction also lets
researchers save deliberately edited result datasets without changing the run.

If evolution is interrupted between requested samples, live time and Br retain
the last completed integration endpoint. Continuation uses that state, not the
older recorded sample. Reopening from disk still starts at the last persisted
dynamic checkpoint.

Input intervals
~~~~~~~~~~~~~~~

``FieldTimeSeries.iter_intervals`` reads adjacent input rows once and supplies
``(left, right, entries)`` with held coefficients. Identical records retain
their entry identity and do not restart an integrator. Independent stream
clocks merge with the same float64-roundoff policy as scalar time lookup.
The final zero-length interval carries inputs exactly at the endpoint:
magnetic state is continuous, while electric field and currents use the new
inputs. No extra lookup or coefficient comparison is needed in evolution.

Each new iteration reads current datasets, including direct edits and removed
streams. Only unchanged selected entries are reused across continuation calls;
there is no persistent input schedule or second history cache.
