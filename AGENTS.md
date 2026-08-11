# PynaMIT development style

Write the numerical code for scientists who read and explore it interactively.

- Prefer a direct sequence of ordinary Python statements. A helper should express a reused
  concept, isolate a genuinely difficult boundary, or make an equation clearer.
- Do not add a class, dataclass, protocol, registry, or wrapper when a function, dictionary,
  array, or direct call says the same thing plainly.
- Validate data at user, file, provider, coordinate, unit, and numerical-solver boundaries.
  Trust values that have already crossed those boundaries; do not repeat defensive checks in
  each internal layer.
- Fail explicitly when required information is missing. Do not guess, silently fall back, or
  catch broad exceptions unless degradation is an intentional documented feature.
- Keep the common IPython workflow short and inspectable. Put specialized machinery behind
  optional or clearly named entry points, not in the path of ordinary simulation use.
- Retain standard scientific symbols when they aid comparison with the equations. Prefer plain,
  descriptive names for software-only concepts.
- Add an abstraction only after identifying the concrete duplication or state ownership it
  removes. Delete forwarding-only and single-use helpers when inlining is clearer.

Preserve scientific behavior and run the relevant numerical tests after refactoring.
