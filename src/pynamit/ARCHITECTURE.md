## Core object model

- `Basis` owns exact forward operators.
  - Examples: evaluation matrices, derivatives, exact vector-basis operators.
  - Use the raw basis when you want basis-family machinery.

- `FieldSpec` owns coefficient-space semantics.
  - It combines `basis + field_type + mean_free`.
  - Use `FieldSpec` when an operation depends on representation details such as:
    - SH `mean_free`
    - scalar vs tangential/vector coefficient layout
    - projection / analysis into coefficient space

- `Field` is the realized field-value facade.
  - It wraps coefficients, sampled values, or an analytic provider behind one frontend.

- `analysis.py` owns backward / fitted / regularized operators.
  - Keep regularization, weighted projection, and least-squares setup out of `Basis`.

- `Timeseries` owns time-indexed storage plus `FieldSpec`-based schema.

- `SimulationData` owns one persisted run directory.
  - One run = one directory with fixed artifact names.

## Mean-free rule

- `mean_free=True` means a zero-mean field.
- SH scalar/tangential spaces realize this by omitting the `(n,m)=(0,0)` coefficient.
- CS scalar spaces realize this through mean-zero constraints/projectors, not by shrinking storage.

## Rule of thumb

- If coefficient-space semantics matter, use `FieldSpec`, not the raw `Basis`.
- If you only need exact forward basis operators, use `Basis`.
