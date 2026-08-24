# PynaMIT development style

Write simulation code for scientists who read, modify, and explore it interactively.

- Put physical quantities, governing equations, units, and coordinate conventions before
  software machinery. In physics-facing functions, make the governing sequence visible by
  composing reusable numerical operations in an order scientists can compare with the
  equations, inspect interactively, and modify locally. Keep physical assumptions, signs,
  units, coordinate conventions, and coupling terms explicit at this level rather than hiding
  them in generic orchestration machinery.
- Build and reuse numerical primitives at the appropriate domain boundary. General spherical
  bases, meshes, projections, transforms, differential operators, linear maps, solvers, and
  backend-neutral array operations belong in Kompe when their meaning is complete without the
  PynaMIT model. Reusable induction, ionospheric-closure, magnetic-boundary, input, evolution,
  and result machinery whose contracts depend on MIT physics belongs in PynaMIT. Do not move
  code to Kompe solely because it is reused.
- Organize modules around cohesive scientific, mathematical, workflow, or infrastructure
  roles, not around line count or one class per file. Split a module when it mixes independently
  understandable calculations or state ownership, but prefer shallow directories and direct
  navigation. Avoid forwarding-only modules, generic utility collections, and fragmented call
  chains. Keep providers, storage, caching, compatibility, and backend handling at clearly
  named boundaries.
- Keep the ordinary IPython workflow short and inspectable: prepare inputs, construct and
  evolve a `Simulation`, and inspect live or saved results. `InputPreparation`, `Simulation`,
  and `SimulationResults` should retain distinct, useful roles; specialized machinery should
  not enter the common path.
- Prefer a direct sequence of ordinary Python statements. A helper should express a reused
  scientific concept, isolate a genuinely difficult boundary, or make an equation clearer.
  Delete forwarding-only and single-use helpers when inlining is easier to follow.
- Use a class or other abstraction when it represents a stable scientific or workflow
  concept, owns meaningful reusable state, isolates a necessary boundary, or removes concrete
  duplication. Do not add a dataclass, protocol, registry, wrapper, or object merely to relay
  calls that a function, dictionary, array, or direct expression states plainly.
- Validate data at user, file, provider, coordinate, unit, and numerical-solver boundaries.
  Trust values that have already crossed those boundaries; do not repeat defensive checks in
  each internal layer.
- Fail explicitly when required information is missing or behavior is unsupported. Do not
  guess, silently fall back, or catch broad exceptions unless degradation is an intentional,
  documented feature.
- Use one canonical name and representation for each physical quantity or operation. Retain
  standard scientific symbols when they aid comparison with equations, and prefer plain,
  descriptive names for software-only concepts. Keep any necessary compatibility alias at a
  public boundary rather than propagating it through the numerical code.
- Name transformations by their mathematical direction. Use projection or analysis for sampled
  fields to coefficients, and evaluation or synthesis for coefficients to field values. Make
  coordinate systems, component order, domains, and codomains explicit at transformation
  boundaries.
- Use the configured shared array backend for pure numerical operations supported by NumPy
  and JAX. Do not convert arrays to NumPy merely to call familiar helpers. Keep SciPy
  algorithms, provider libraries, file I/O, and plotting at explicit CPU boundaries, and make
  unavoidable host-device transfers visible.
- Preserve useful operator structure such as diagonal, sparse, or matrix-free forms. Materialize
  an operator only when repeated use benefits from it, reuse that materialization afterward,
  and do not turn a diagonal vector into a dense matrix without a numerical reason.
- Test physical identities, coordinate and unit conventions, numerical equivalence, and
  complete researcher-facing workflows. Exercise portable numerical paths with both NumPy and
  JAX; keep native-provider equivalence and performance checks focused instead of multiplying
  the full simulation suite.

Preserve scientific behavior and numerical performance, and run the relevant tests after
changes.
