# Test suite

The test tree follows scientific responsibilities rather than generic labels
such as “unit” and “integration”:

- `regression/` contains the small, one-scenario numerical regressions.
- `inputs/` covers empirical providers, input contracts, projection, and
  prepared input packages.
- `mage/` covers MAGE forcing preparation, projection, and runnable scripts.
- `visualization/` covers result inspection and plotting.
- The root contains the core simulation, field, geometry, storage, and backend
  tests.

The normal suite runs on NumPy and, when installed, JAX using the bundled input
snapshot. Native Lompe, PyAMPS, and HWM validation is deliberately focused:

```bash
pytest -q
pytest -q --backend numpy --data-source fallback
pytest -q -m native_input_validation --backend numpy --backend jax --data-source native
```

The normal suite also exercises every supported least-squares algorithm on the
same small electrodynamic reconstruction problem. This gives each algorithm a
direct physical correctness check without rerunning the entire suite five times.
The deterministic baseline for unrelated tests remains `normal_pinv`.
For a broader diagnostic run with one alternative solver, use for example
`pytest -q --least-squares-solver lsmr`.

Keep numerical regression scenarios in separate files when that makes their
physical configuration easier to read. Share helpers only when they represent
a substantial reused setup, such as constructing a complete synthetic MAGE
forcing file.

`example_scenario.py` states the event time and every physical input passed to
Hardy, AMPS, and HWM by the shared regression case. Those values belong to the
tests; PynaMIT's empirical-input workflow requires callers to provide them.
