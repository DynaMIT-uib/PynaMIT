Time-integration benchmarks
===========================

These benchmarks separate numerical construction, first use, cached application,
and end-to-end evolution. They are measurements, not CI performance limits or GPU
speedup claims. Euler remains the default integrator.

Directly forced exponential
---------------------------

The equation is ``x' = A x + b``. ``kompe.math.affine_exponential`` returns
``P, q`` such that ``x(t+h) = P(x(t)) + q``. Dense construction uses an
``(n+1) x (n+1)`` augmented matrix, rather than a ``2n x 2n`` full
forcing-response matrix. It requires no equilibrium solve or inverse of A,
including for forced null modes. Large forcing units are handled by an exact
rescaling of the auxiliary coordinate, not by shifting eigenvalues.

The updated ``scripts/tools/benchmark_exponential_blocks.py`` compares:

* Construction of the old homogeneous exponential ``exp(h*A)``.
* Construction of the directly forced ``P, q``.
* Warm production application with a cached duration.
* Production construction and stepping after the forcing changes.
* A benchmark-only full forcing response, ``x_next = P*x + Q*b``,
  built from a ``2n x 2n`` exponential and reused across changed forcing.

An independent equilibrium-centred reference checks the same fixed equation
where an equilibrium exists. Singular and non-normal equations without an
equilibrium are tested separately against manufactured solutions.

The example uses nonuniform conductance,
``SigmaP = 5 + cos(theta)`` and ``SigmaH = 3 + sin(theta)*cos(phi)`` siemens,
a dipole main field, no PFAC/interhemispheric coupling, and ``normal_pinv``.
SH cases use ``Nmax=Mmax`` of 5 and 20 with ``Ncs=12``.

Measured on 8 September 2026, macOS arm64 CPU, Python 3.12.13, NumPy 2.4.3,
SciPy 1.17.1, JAX 0.9.2, float64. The duration is 0.2 s. Values are median
milliseconds from five calls after compilation; setup is reported separately.

.. list-table:: Exponential construction and application
   :header-rows: 1

   * - Backend
     - Coefficients
     - Homogeneous expm
     - Forced expm
     - Cached production step
     - Changed-forcing step
   * - NumPy
     - 35
     - 0.104
     - 0.144
     - 0.028
     - 0.175
   * - NumPy
     - 440
     - 8.00
     - 8.81
     - 0.090
     - 8.84
   * - JAX CPU
     - 35
     - 0.227
     - 0.628
     - 0.129
     - 0.757
   * - JAX CPU
     - 440
     - 6.20
     - 6.81
     - 0.255
     - 6.94

The augmented construction is not intrinsically faster than homogeneous expm.
It avoids the separate equilibrium fit when initialization and equilibrium
output do not request that fit. In the 440-coefficient example, constructing
the equilibrium diagnostic took about 52 ms on NumPy and 190 ms on JAX CPU
(the latter includes first-use compilation). Subsequent equilibrium applications
can reuse their prepared solve; those setup times are not per-step savings.

The largest relative difference from the equilibrium-centred reference in these
runs was ``1.26e-13``. JAX's first affine construction cost roughly 0.34--0.36 s;
warm construction must not be mistaken for first-use latency.

The production stepper retains four recent duration maps for fixed conductance
and forcing. Changing forcing invalidates their increments and requires new
construction. A regular requested cadence is reused despite timestamp
subtraction roundoff. Irregular durations may require more matrix exponentials.
There is no hardware-dependent crossover heuristic or adaptive inner solve.
Each cached dense float64 map occupies about ``8*n*(n+1)`` bytes plus small
array-view overhead; diagonal generators retain vectors instead.

Full forcing-response reuse
---------------------------

The additional benchmark holds both the operator and duration fixed while
changing the source vector. On the same CPU, five warm measurements at
440 coefficients and 0.2 s gave:

.. list-table:: Fixed-operator, changed-forcing comparison (milliseconds)
   :header-rows: 1

   * - Backend
     - Current changed-forcing step
     - Full response construction
     - Full response application
   * - NumPy
     - 9.44
     - 43.35
     - 0.087
   * - JAX CPU
     - 7.91
     - 25.88
     - 0.178

Warm construction amortizes after about five NumPy or four JAX forcing vectors.
The extra retained float64 matrix costs approximately 1.55 MB at this size;
construction needs a larger temporary matrix. JAX's first full construction
took 200 ms including compilation, so the warm crossover is not a first-run
crossover. The full-response application assumes an already assembled forcing
vector and is an optimistic reuse estimate, not a complete workflow timing.

This remains benchmark-only. Changes in conductance generally change A, and
different durations need different responses. A typical MAGE sequence changing
both conductance and forcing cannot reuse this matrix across those changes.
For deliberate fixed-conductance experiments with many source changes it is
promising, but it is not a reason to add automatic solver-selection heuristics.
The benchmark checks each changed-forcing result against production propagation.

End-to-end evolution
--------------------

``scripts/tools/benchmark_evolution.py`` includes forcing selection, integration,
physical output-field calculation, and in-memory xarray updates. It excludes
disk I/O and native providers. Each row below is a median over five warm
continuations after first use.

.. list-table:: Current sampled evolution
   :header-rows: 1

   * - Scenario
     - NumPy (ms)
     - JAX CPU (ms)
   * - Euler, 35 coefficients, 0.2 s duration, dt=0.0002 s, output every 0.002 s
     - 34.7
     - 65.7
   * - Exponential, 440 coefficients, 2 s duration, output every 0.2 s
     - 5.6
     - 9.6

Output batches contain ten samples. A single input record is held throughout
each example; input changes are not skipped. Earlier measurements of the
previous block integrator were approximately 40/72 ms for the Euler case and
7/13 ms for the exponential case on NumPy/JAX CPU. Differences of a few
milliseconds on these small workloads should not be treated as universal
speedups. First-use JAX evolution still takes roughly two seconds.

Output spacing is independent of internal stepping. Euler retains fixed
accepted steps and uses a linear within-step interpolant. SciPy adapts across
each constant-input interval, with dense output rather than a solver restart at
every requested sample. Its accepted-step count is tested to be independent of
output frequency. Exponential evolution applies exact affine maps between
outputs and physical input boundaries; it has no artificial internal dt.

Reproduce the comparisons with::

    python scripts/tools/benchmark_exponential_blocks.py --backend numpy \
        --degrees 5 20 --durations 0.2 --repeats 5
    JAX_ENABLE_X64=1 python scripts/tools/benchmark_exponential_blocks.py \
        --backend jax --degrees 5 20 --durations 0.2 --repeats 5

    python scripts/tools/benchmark_evolution.py --backend numpy --repeats 5
    JAX_ENABLE_X64=1 python scripts/tools/benchmark_evolution.py \
        --backend jax --repeats 5

    python scripts/tools/benchmark_evolution.py --backend numpy \
        --integrator exponential --degree 20 --steps 10000 --sample 1000 --repeats 5

The end-to-end benchmark's ``steps`` and ``sample`` counts are convenience
multipliers for its base ``dt``: they specify duration and output interval.
They are not internal steps of the exponential integrator. Add
``--horizontal CS --degrees 5 --ncs 8`` to the construction benchmark for a
CS horizontal case. Production MAGE data and the intended accelerator still
need workload-specific measurements.

Live-state and input-interval ownership
----------------------------------------

On 9 September 2026, the same script compared a source snapshot before the
live-state/interval change against the updated implementation. macOS arm64 CPU,
Python 3.12.13, float64, single-threaded Accelerate; medians of 11 warm
continuations, excluding setup and first-use compilation:

.. list-table:: Warm continuation, milliseconds
   :header-rows: 1

   * - Case
     - Before
     - After
   * - NumPy, Euler, 35 coefficients, 100 output samples
     - 18.8
     - 18.6
   * - JAX CPU, Euler, 35 coefficients, 100 output samples
     - 28.0
     - 25.7
   * - NumPy, exponential, 440 coefficients, 10 output samples
     - 4.24
     - 3.84
   * - JAX CPU, exponential, 440 coefficients, 10 output samples
     - 5.54
     - 5.33

These use the Euler defaults and exponential command above, respectively,
with ``--repeats 11``. Sample counts and final magnetic-field norms agreed
exactly within each backend. These small timings support preserved performance,
not a general speedup claim. The primary changes are independent live state and
single-pass input selection; stepping algorithms are unchanged.

Bounded evolution sample blocks
-------------------------------

On 10 September 2026, the end-to-end benchmark compared the pre-change source
snapshot with bounded numerical output blocks. The machine was macOS 26.6.2
arm64, Python 3.12.13, float64, with ``VECLIB_MAXIMUM_THREADS=1``. Each value
is the median of 11 warm continuations; no tests or other benchmarks ran
concurrently. The parameters match the preceding table, with ten samples per
write. Setup, first-use compilation, provider calls, and disk I/O are excluded.

.. list-table:: Warm continuation, milliseconds
   :header-rows: 1

   * - Case
     - Before
     - After
   * - NumPy, Euler, 35 coefficients, 100 output samples
     - 19.85
     - 19.63
   * - JAX CPU, Euler, 35 coefficients, 100 output samples
     - 27.38
     - 21.18
   * - NumPy, exponential, 440 coefficients, 10 output samples
     - 4.22
     - 4.13
   * - JAX CPU, exponential, 440 coefficients, 10 output samples
     - 5.85
     - 4.96

Output counts and final magnetic-field norms agreed exactly within each
backend. The measured JAX improvements are approximately 23% and 15%; the
NumPy differences are small enough to treat as essentially unchanged.
First-use JAX evolution remained about 2.2--2.6 seconds. These are CPU results,
not a claim about GPU speedups or arbitrary workloads.

The implementation batches numerical samples and physical output evaluation,
not independent trajectories with different operators. Internal Euler steps,
adaptive solver state, diagonal structure, and exact input-change boundaries
are unchanged.
