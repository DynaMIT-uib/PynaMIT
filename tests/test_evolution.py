"""Focused tests for evolution scheduling and propagator reuse."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from kompe.math import LinearMap, as_linear_map, get_array_module, linear_evolution

from pynamit.simulation.electrodynamics import induction
from pynamit.simulation.evolution import _TimeEvolution
from pynamit.simulation.response import ElectrodynamicResponse


def _linear_samples(Br, times, *, batch_size, **_sampling):
    """Manufactured dBr/dt=1, sampled in the requested blocks."""
    for start in range(0, len(times), batch_size):
        yield Br[..., None] + times[start : start + batch_size]


class _FakeResponse:
    def __init__(self):
        self._conductance_values = None
        self.induced_poloidal_potential_feedback_operator = as_linear_map(np.eye(1))
        identity = as_linear_map(np.eye(1))
        self.geometry = SimpleNamespace(
            main_field=SimpleNamespace(kind="radial"),
            induced_poloidal_potential_faraday_rate_scale=1.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
            helmholtz_divergence_free_potential_operator=identity,
            surface_to_poloidal_operator=identity,
        )

    @staticmethod
    def solve_noninductive_response():
        return np.zeros(1), np.zeros(1)


class _FakeSimulation:
    def __init__(self, *, integrator="euler"):
        self.config = SimpleNamespace(
            integrator=integrator, save_equilibria=False, enable_pfac_coupling=False
        )
        self._response = _FakeResponse()
        self.response.config = self.config
        self.geometry = self.response.geometry
        self.current_time = np.float64(0.0)
        self.induced_Br = None
        self.recorded = []
        self.saved = []
        self.inputs = {}

        def save_output(key, _store):
            self.saved.append(key)

        output_series = SimpleNamespace(datasets={}, save=save_output)
        self.results = SimpleNamespace(
            input_series=SimpleNamespace(
                variables={"conductance": ()},
                get_field_space=lambda _key: None,
                iter_intervals=lambda start, stop, **_kwargs: iter(
                    [(start, stop, {"conductance": None}), (stop, stop, {"conductance": None})]
                ),
            ),
            output_series=output_series,
            artifact_store=SimpleNamespace(),
            schema=SimpleNamespace(
                output_field_spaces={"dynamic": {"induced_Br": SimpleNamespace(shape=(1,))}}
            ),
        )

    @property
    def response(self):
        return self._response

    def _save_output(self, *keys):
        for key in keys:
            self.results.output_series.save(key, self.results.artifact_store)

    @response.setter
    def response(self, response):
        response._conductance_values = None
        response.config = self.config
        self._response = response


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan])
def test_evolution_rejects_invalid_time_step(dt):
    """Invalid time steps fail before entering the run loop."""
    simulation = _FakeSimulation()
    with pytest.raises(ValueError, match="dt must be finite and greater than zero"):
        _TimeEvolution(simulation).evolve_to_time(1.0, dt=dt, quiet=True)


def test_progress_reports_physical_output_times(monkeypatch, capsys):
    """Progress is about output time, not hidden solver step counts."""
    evolution = _TimeEvolution(_FakeSimulation())
    monkeypatch.setattr(induction, "equilibrium_induced_Br", lambda *_args, **_kwargs: np.zeros(1))
    monkeypatch.setattr(evolution, "_record_output_snapshot", lambda *_args, **_kwargs: None)
    evolution.evolve_to_time(
        1e-4, dt=1e-8, output_interval=1e-4, samples_per_write=1, sample_equilibrium=True
    )
    assert "Output at t = 0 / 0.0001 s" in capsys.readouterr().out


@pytest.mark.parametrize("value", [0, -1, np.inf, True])
def test_evolution_rejects_invalid_sample_intervals(value):
    """The output interval is a positive finite physical duration."""
    simulation = _FakeSimulation()
    with pytest.raises(ValueError, match="output_interval"):
        _TimeEvolution(simulation).evolve_to_time(1.0, output_interval=value, quiet=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"t": True}, "t"),
        ({"t": 1.0, "dt": False}, "dt"),
        ({"t": 1.0, "quiet": "false"}, "quiet"),
        ({"t": 1.0, "sample_equilibrium": "true"}, "sample_equilibrium"),
    ],
)
def test_evolution_rejects_ambiguous_runtime_option_types(kwargs, match):
    """Reject options with ambiguous truth meaning."""
    simulation = _FakeSimulation()

    with pytest.raises(ValueError, match=match):
        _TimeEvolution(simulation).evolve_to_time(**kwargs)


def test_evolution_rejects_backfill_from_later_checkpoint():
    """A later checkpoint cannot generate earlier missing output."""
    simulation = _FakeSimulation()
    simulation.current_time = 10.0
    simulation.induced_Br = np.zeros(1)
    simulation.results.output_series = SimpleNamespace(
        datasets={"dynamic": xr.Dataset(coords={"time": [10.0]})},
        get_entry=lambda *_args, **_kwargs: {"induced_Br": np.zeros(1)},
    )

    with pytest.raises(ValueError, match="precedes the active checkpoint"):
        _TimeEvolution(simulation).evolve_to_time(5.0, sample_equilibrium=True, quiet=True)


def test_evolution_records_and_saves_exact_off_grid_target(monkeypatch):
    """An off-grid target is still a final checkpoint."""
    simulation = _FakeSimulation()

    monkeypatch.setattr(
        induction,
        "build_induction_stepper",
        lambda _response, _forcing, **_options: _linear_samples,
    )

    evolution = _TimeEvolution(simulation)
    monkeypatch.setattr(
        evolution,
        "_record_output_snapshot",
        lambda key, *_values, time: simulation.recorded.append((key, time)),
    )
    evolution.evolve_to_time(
        0.25,
        dt=0.1,
        output_interval=1.0,
        samples_per_write=10,
        initialize_from_equilibrium=False,
        sample_equilibrium=False,
        quiet=True,
    )

    assert simulation.current_time == np.float64(0.25)
    assert simulation.recorded == [("dynamic", 0.0), ("dynamic", 0.25)]
    assert simulation.saved == ["dynamic", "dynamic"]


@pytest.mark.parametrize("sample_equilibrium", [False, True])
def test_evolution_sampling_and_writing_share_the_requested_streams(
    sample_equilibrium, monkeypatch
):
    """Advance dBr/dt=1 and sample equilibrium on the same schedule."""
    simulation = _FakeSimulation()
    monkeypatch.setattr(
        induction,
        "build_induction_stepper",
        lambda _response, _forcing, **_options: _linear_samples,
    )
    monkeypatch.setattr(
        induction, "equilibrium_induced_Br", lambda *_args, **_kwargs: np.array([10.0])
    )
    samples = {"dynamic": [], "equilibrium": []}

    def record(key, induced_Br, *_args, time):
        times = np.atleast_1d(time)
        values = np.broadcast_to(np.asarray(induced_Br).reshape(-1), times.shape)
        samples[key].extend(zip(times, values, strict=True))

    evolution = _TimeEvolution(simulation)
    monkeypatch.setattr(evolution, "_record_output_snapshot", record)
    evolution.evolve_to_time(
        0.55,
        dt=0.1,
        output_interval=0.2,
        samples_per_write=2,
        initialize_from_equilibrium=False,
        sample_equilibrium=sample_equilibrium,
        quiet=True,
    )

    assert simulation.current_time == 0.55
    for key, enabled, offset in (
        ("dynamic", True, 0.0),
        ("equilibrium", sample_equilibrium, 10.0),
    ):
        writes = [stream for stream in simulation.saved if stream == key]
        if enabled:
            times, values = np.asarray(samples[key]).T
            np.testing.assert_allclose(times, [0.0, 0.2, 0.4, 0.55])
            np.testing.assert_allclose(values, times if key == "dynamic" else offset)
            assert writes == [key] * 3
        else:
            assert samples[key] == []
            assert writes == []


def test_evolution_does_not_rewrite_outputs_that_already_reach_target(monkeypatch):
    """Reopening a completed trajectory does not solve or save again."""
    simulation = _FakeSimulation()
    simulation.current_time = 0.5
    simulation.induced_Br = np.ones(1)
    simulation.results.output_series.datasets = {
        key: xr.Dataset(coords={"time": [0.0, 0.5]}) for key in ("dynamic", "equilibrium")
    }
    simulation.results.output_series.get_entry = lambda *_args, **_kwargs: {
        "induced_Br": np.ones(1)
    }

    def unexpected_solve():
        pytest.fail("Completed output should not require an electrodynamic solve.")

    monkeypatch.setattr(simulation.response, "solve_noninductive_response", unexpected_solve)
    _TimeEvolution(simulation).evolve_to_time(0.4, sample_equilibrium=True, quiet=True)

    assert simulation.current_time == 0.5
    assert simulation.saved == []


@pytest.mark.parametrize("integrator", ["euler", "exponential", "RK45"])
@pytest.mark.parametrize("materialized", [False, True])
def test_block_kernel_preserves_linear_map_execution(integrator, materialized):
    """Structured and explicit maps give the same repeated steps."""
    xp = get_array_module()
    feedback = LinearMap(
        shape=(1, 1), dtype=float, matvec=lambda values: -values, rmatvec=lambda values: -values
    )
    if materialized:
        feedback.to_matrix()
    identity = as_linear_map(xp.eye(1))
    response = SimpleNamespace(
        config=SimpleNamespace(integrator=integrator),
        induced_poloidal_potential_feedback_operator=feedback,
        geometry=SimpleNamespace(
            induced_poloidal_potential_faraday_rate_scale=1.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
            helmholtz_divergence_free_potential_operator=identity,
            surface_to_poloidal_operator=identity,
        ),
    )
    dt, steps = 0.1, 7
    result = induction.evolve_induced_Br(
        response,
        xp.array([2.0]),
        dt * steps,
        xp.ones(1),
        dt=dt if integrator == "euler" else None,
        rtol=1e-10,
        atol=1e-12,
    )
    decay = (1 - dt) ** steps if integrator == "euler" else np.exp(-steps * dt)
    np.testing.assert_allclose(result, 1 + decay, rtol=1e-8)
    if xp is not np:
        assert "jax" in type(result).__module__
    # Euler itself has no numerical reason to materialize the map.
    if not materialized and integrator == "euler":
        assert not feedback._dense_cache


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize("kind", ["diagonal", "dense", "exponential"])
def test_changed_responses_reuse_compiled_array_steps(backend, data_source, kind, monkeypatch):
    """Changed same-shaped coefficients do not recompile stepping."""
    from kompe.math import diagonal_linear_map, identity_linear_map

    xp = get_array_module()
    identity = identity_linear_map((2,))
    geometry = SimpleNamespace(
        induced_poloidal_potential_faraday_rate_scale=1.0,
        induced_Br_to_poloidal_potential_operator=identity,
        induced_poloidal_potential_to_Br_operator=identity,
        helmholtz_divergence_free_potential_operator=identity,
        surface_to_poloidal_operator=identity,
    )
    name = "_affine_samples" if kind == "exponential" else "_euler_samples"
    original = getattr(linear_evolution, name)
    traces = []

    def traced(*args, **kwargs):
        traces.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(linear_evolution, name, traced)
    for value in (1.0, 2.0, 3.0):
        rates = value * xp.array([1.0, 2.0])
        feedback = (
            diagonal_linear_map(-rates) if kind == "diagonal" else as_linear_map(-xp.diag(rates))
        )
        response = SimpleNamespace(
            config=SimpleNamespace(integrator="exponential" if kind == "exponential" else "euler"),
            geometry=geometry,
            induced_poloidal_potential_feedback_operator=feedback,
        )
        equilibrium = 1 / rates
        result = induction.evolve_induced_Br(
            response,
            xp.array([2.0, 3.0]),
            0.1,
            xp.ones(2),
            dt=0.01 if kind != "exponential" else None,
        )
        decay = xp.exp(-0.1 * rates) if kind == "exponential" else (1 - 0.01 * rates) ** 10
        expected = equilibrium + (xp.array([2.0, 3.0]) - equilibrium) * decay
        np.testing.assert_allclose(result, expected, rtol=1e-13)
        if kind == "diagonal":
            assert not feedback._dense_cache
    assert len(traces) == 1


def test_scipy_step_uses_feedback_operator():
    """Accept a structured feedback operator at the SciPy boundary."""
    identity = as_linear_map(np.eye(1))
    response = SimpleNamespace(
        config=SimpleNamespace(integrator="RK45"),
        induced_poloidal_potential_feedback_operator=as_linear_map(np.array([[-1.0]])),
        geometry=SimpleNamespace(
            induced_poloidal_potential_faraday_rate_scale=1.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
            helmholtz_divergence_free_potential_operator=identity,
            surface_to_poloidal_operator=identity,
        ),
    )

    evolved = induction.evolve_induced_Br(response, np.ones(1), 0.1, np.zeros(1))

    np.testing.assert_allclose(evolved, np.exp(-0.1), rtol=1e-6)


@pytest.mark.parametrize("materialized", [False, True])
def test_induction_does_not_modify_state_when_operator_returns_a_view(materialized):
    """An operator may return a view of its input."""
    from kompe.math import identity_linear_map

    xp = get_array_module()
    identity = identity_linear_map((2,))
    feedback = LinearMap(
        shape=(2, 2), dtype=float, matvec=lambda values: values, rmatvec=lambda values: values
    )
    if materialized:
        feedback.to_matrix()
    response = SimpleNamespace(
        config=SimpleNamespace(integrator="euler"),
        induced_poloidal_potential_feedback_operator=feedback,
        geometry=SimpleNamespace(
            induced_poloidal_potential_faraday_rate_scale=1.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
            helmholtz_divergence_free_potential_operator=identity,
            surface_to_poloidal_operator=identity,
        ),
    )
    initial = xp.asarray([2.0, 3.0])
    forcing = xp.asarray([1.0, -2.0])
    derivative = induction.induced_Br_time_derivative(response, initial, forcing)
    np.testing.assert_array_equal(initial, [2.0, 3.0])
    np.testing.assert_allclose(derivative, [3.0, 1.0])
    result = induction.evolve_induced_Br(response, initial, 0.3, forcing, dt=0.1)
    np.testing.assert_array_equal(initial, [2.0, 3.0])
    np.testing.assert_array_equal(forcing, [1.0, -2.0])
    np.testing.assert_allclose(result, (np.array([3.0, 1.0]) * 1.1**3) - [1.0, -2.0])


@pytest.mark.parametrize("integrator", ["euler", "exponential", "RK45"])
@pytest.mark.parametrize("radius", [6.5e6, 8.0e6])
def test_uniform_radial_field_follows_analytic_harmonic_relaxation(integrator, radius):
    """Check Hall forcing, Pedersen decay, and physical scaling."""
    from kompe.constants import MU0

    from pynamit import Simulation

    simulation = Simulation(
        Nmax=3,
        Mmax=2,
        Ncs=8,
        RI=radius,
        main_field_kind="radial",
        enable_pfac_coupling=False,
        integrator=integrator,
    )
    grid = simulation.model_grid
    simulation.inputs.set_conductance(
        pedersen=np.full(grid.size, 5.0), hall=np.full(grid.size, 3.0), grid=grid
    )
    response = simulation.response
    degree = simulation.geometry.poloidal_basis.n
    jr = np.linspace(-1e-6, 1e-6, degree.size)
    electric_forcing, _ = response.solve_noninductive_response(boundary_jr=jr)
    # For outward radial B, Hall currents drive negative Br while
    # Pedersen resistance gives each harmonic its own positive decay.
    equilibrium = -MU0 * radius * (3.0 / 5.0) * jr / (2 * degree + 1)
    rate = (5.0 / 34.0) * (2 * degree + 1) / (MU0 * radius)
    np.testing.assert_allclose(
        induction.equilibrium_induced_Br(response, electric_forcing), equilibrium, atol=1e-18
    )
    dt, steps = 0.01, 10
    xp = get_array_module()
    actual = induction.evolve_induced_Br(
        response,
        xp.zeros(degree.size),
        dt * steps,
        electric_forcing,
        dt=dt if integrator == "euler" else None,
        rtol=1e-11,
        atol=1e-13,
    )
    decay = (1 - dt * rate) ** steps if integrator == "euler" else np.exp(-dt * steps * rate)
    np.testing.assert_allclose(actual, equilibrium * (1 - decay), rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("integrator", ["euler", "exponential"])
@pytest.mark.parametrize("steps", [1, 37, 1000])
def test_induction_blocks_match_coupled_manufactured_solution(integrator, steps):
    """Check coupled dynamics with nonuniform Br conversion."""
    from kompe.math import diagonal_linear_map

    xp = get_array_module()
    feedback = xp.asarray([[-1.0, 2.0], [0.0, -2.0]])
    Br_scale = xp.asarray([3.0, 7.0])
    identity = as_linear_map(xp.eye(2))
    geometry = SimpleNamespace(
        induced_poloidal_potential_faraday_rate_scale=1.0,
        induced_Br_to_poloidal_potential_operator=diagonal_linear_map(1 / Br_scale),
        induced_poloidal_potential_to_Br_operator=diagonal_linear_map(Br_scale),
        helmholtz_divergence_free_potential_operator=identity,
        surface_to_poloidal_operator=identity,
    )
    response = SimpleNamespace(
        config=SimpleNamespace(integrator=integrator),
        geometry=geometry,
        induced_poloidal_potential_feedback_operator=as_linear_map(feedback),
    )
    dt = 0.01
    equilibrium_potential = xp.asarray([1.0, 2.0])
    initial_potential = xp.asarray([4.0, -1.0])
    forcing = -feedback @ equilibrium_potential
    actual = induction.evolve_induced_Br(
        response,
        Br_scale * initial_potential,
        dt * steps,
        forcing,
        dt=dt if integrator == "euler" else None,
    )
    # For this triangular system, both propagators have a closed form.
    a = (1 - dt) ** steps if integrator == "euler" else np.exp(-dt * steps)
    b = (1 - 2 * dt) ** steps if integrator == "euler" else np.exp(-2 * dt * steps)
    propagator = np.array([[a, 2 * (a - b)], [0.0, b]])
    expected = np.array([3.0, 7.0]) * (np.array([1.0, 2.0]) + propagator @ [3.0, -3.0])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
@pytest.mark.parametrize("steps", [1, 17])
def test_scipy_step_transfers_only_completed_forcing(backend, data_source, steps, monkeypatch):
    """Finish forcing on JAX before crossing into SciPy."""
    import jax
    import jax.numpy as jnp
    from kompe.math import LinearMap

    transfers = []
    original_to_numpy = linear_evolution.to_numpy

    def to_numpy(values):
        transfers.append(values)
        return original_to_numpy(values)

    def surface_to_poloidal(values):
        assert isinstance(values, jax.Array)
        return 2.0 * values

    identity = as_linear_map(jnp.eye(1))
    response = SimpleNamespace(
        config=SimpleNamespace(integrator="RK45"),
        induced_poloidal_potential_feedback_operator=as_linear_map(jnp.array([[-1.0]])),
        geometry=SimpleNamespace(
            induced_poloidal_potential_faraday_rate_scale=1.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
            helmholtz_divergence_free_potential_operator=identity,
            surface_to_poloidal_operator=LinearMap(
                shape=(1, 1), dtype=float, matvec=surface_to_poloidal, rmatvec=surface_to_poloidal
            ),
        ),
    )
    monkeypatch.setattr(linear_evolution, "to_numpy", to_numpy)
    evolved = induction.evolve_induced_Br(
        response, jnp.array([2.0]), 0.1 * steps, jnp.array([0.5]), rtol=1e-10, atol=1e-12
    )

    assert len(transfers) == 3  # Forcing, tolerance conversion, initial state.
    assert isinstance(evolved, jax.Array)
    np.testing.assert_allclose(evolved, 1.0 + np.exp(-0.1 * steps), rtol=1e-6)


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["jax"], ids=["backend=jax"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_output_snapshot_stays_on_jax_until_storage_boundary(backend, data_source):
    """Keep output snapshots on JAX until the storage boundary."""
    import jax.numpy as jnp

    captured = {}
    geometry = SimpleNamespace(
        horizontal_basis=SimpleNamespace(project_helmholtz_mean_free=lambda values: values)
    )
    response = SimpleNamespace(
        geometry=geometry,
        solve_induced_response=lambda induced: (
            jnp.zeros((2, *induced.shape)),
            jnp.zeros_like(induced),
        ),
    )
    response.output_coefficients = ElectrodynamicResponse.output_coefficients.__get__(response)
    simulation = SimpleNamespace(
        response=response,
        geometry=geometry,
        current_time=np.float64(0.0),
        results=SimpleNamespace(
            output_series=SimpleNamespace(
                add_entries=lambda key, values, times: captured.update(values)
            )
        ),
    )

    evolution = _TimeEvolution(simulation)
    evolution._record_output_snapshot(
        "dynamic", jnp.ones(1), jnp.ones((2, 1)), jnp.ones(1), response, time=0
    )
    assert not captured
    evolution._flush_output_samples()

    assert set(captured) == {"induced_Br", "boundary_jr", "Phi", "W"}
    assert all("jax" in type(values).__module__ for values in captured.values())


def test_output_batches_share_response_without_sharing_forcing():
    """Batch held responses without changing individual inputs."""
    xp = get_array_module()
    calls = []
    stored = []

    def induced_response(scale):
        def solve(induced_Br):
            calls.append((scale, induced_Br.shape))
            E = xp.stack((scale * induced_Br, -scale * induced_Br))
            return E, 2 * scale * induced_Br

        response = SimpleNamespace(
            solve_induced_response=solve,
            geometry=SimpleNamespace(
                horizontal_basis=SimpleNamespace(project_helmholtz_mean_free=lambda values: values)
            ),
        )
        response.output_coefficients = ElectrodynamicResponse.output_coefficients.__get__(response)
        return response

    first, second = induced_response(2), induced_response(3)
    simulation = SimpleNamespace(
        geometry=SimpleNamespace(
            horizontal_basis=SimpleNamespace(project_helmholtz_mean_free=lambda values: values)
        ),
        results=SimpleNamespace(
            output_series=SimpleNamespace(add_entries=lambda *args: stored.append(args))
        ),
    )
    evolution = _TimeEvolution(simulation)
    # A response recurs after a change; preserve chronological order.
    scales = [2, 2, 3, 3, 2]
    for time, response in enumerate((first, first, second, second, first)):
        simulation.current_time = time
        evolution._record_output_snapshot(
            "dynamic",
            xp.array([time + 1.0]),
            xp.array([[10.0 * time], [20.0 * time]]),
            xp.array([30.0 * time]),
            response,
            time=time,
        )
    assert not calls and not stored
    evolution._flush_output_samples()
    assert calls == [(2, (1, 2)), (3, (1, 2)), (2, (1, 1))]
    assert len(stored) == 1
    key, fields, times = stored[0]
    assert key == "dynamic"
    np.testing.assert_array_equal(times, list(range(5)))
    time = np.arange(5.0)[:, None]
    induced = np.array(scales)[:, None] * (time + 1)
    np.testing.assert_array_equal(fields["induced_Br"], time + 1)
    np.testing.assert_array_equal(fields["Phi"], 10 * time + induced)
    np.testing.assert_array_equal(fields["W"], 20 * time - induced)
    np.testing.assert_array_equal(fields["boundary_jr"], 30 * time + 2 * induced)
    assert not evolution._output_samples


def _scalar_response(integrator, rate=-1.0):
    """Build x' = rate*x + b in identity coordinates."""
    from kompe.math import diagonal_linear_map, identity_linear_map

    identity = identity_linear_map((1,))
    return SimpleNamespace(
        config=SimpleNamespace(integrator=integrator),
        induced_poloidal_potential_feedback_operator=diagonal_linear_map(
            get_array_module().array([rate])
        ),
        geometry=SimpleNamespace(
            induced_poloidal_potential_faraday_rate_scale=1.0,
            induced_Br_to_poloidal_potential_operator=identity,
            induced_poloidal_potential_to_Br_operator=identity,
            helmholtz_divergence_free_potential_operator=identity,
            surface_to_poloidal_operator=identity,
        ),
    )


def test_scipy_tolerances_refer_to_physical_Br(monkeypatch):
    """Rescale absolute error bounds with the solver coordinates."""
    from kompe.math import diagonal_linear_map

    response = _scalar_response("RK45")
    response.geometry.induced_Br_to_poloidal_potential_operator = diagonal_linear_map(
        get_array_module().array([1 / 6])
    )
    response.geometry.induced_poloidal_potential_to_Br_operator = diagonal_linear_map(
        get_array_module().array([6.0])
    )
    original = linear_evolution.scipy.integrate.RK45
    options = []

    def solver(*args, **kwargs):
        options.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(linear_evolution.scipy.integrate, "RK45", solver)
    induction.evolve_induced_Br(response, np.array([1e-8]), 0.2, np.zeros(1))
    assert options[0]["rtol"] == 1e-3
    np.testing.assert_allclose(options[0]["atol"], [1e-12 / 6], rtol=1e-14, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output_interval": 0.1, "output_times": [0.1]},
        {"output_times": [0.2, 0.1]},
        {"output_times": [-0.1]},
        {"output_times": [0.1, 0.1]},
        {"output_times": [1.1]},
        {"output_times": [np.nan]},
        {"output_times": [[0.1]]},
        {"output_times": [True]},
        {"rtol": 0},
        {"atol": -1},
    ],
)
def test_invalid_output_times_and_integration_tolerances_fail_at_boundary(kwargs):
    """Reject unsupported requests before numerical work."""
    with pytest.raises(ValueError):
        _TimeEvolution(_FakeSimulation()).evolve_to_time(1.0, **kwargs)


@pytest.mark.parametrize("integrator", ["exponential", "RK45"])
def test_dt_is_only_an_euler_control(integrator):
    """Do not silently ignore an irrelevant integration control."""
    with pytest.raises(ValueError, match="dt controls Euler"):
        _TimeEvolution(_FakeSimulation(integrator=integrator)).evolve_to_time(1.0, dt=0.1)


@pytest.mark.parametrize("duration", [0.735, 3e-7])
def test_explicit_final_only_output_still_integrates(duration, monkeypatch):
    """Integrate all internal steps even with one requested output."""
    simulation = _FakeSimulation()
    simulation.response = _scalar_response("euler", rate=0)
    simulation.response.geometry.main_field = SimpleNamespace(kind="radial")
    simulation.geometry = simulation.response.geometry
    simulation.response.solve_noninductive_response = lambda: (np.ones(1), np.zeros(1))
    evolution = _TimeEvolution(simulation)
    values = []
    monkeypatch.setattr(
        evolution, "_record_output_snapshot", lambda key, Br, *_, time: values.append(Br)
    )
    evolution.evolve_to_time(
        duration, dt=0.1, output_times=[duration], quiet=True, initialize_from_equilibrium=False
    )
    np.testing.assert_allclose(values, [[duration]], atol=1e-14)


@pytest.mark.parametrize("integrator", ["euler", "exponential", "RK45"])
def test_forcing_changes_at_its_timestamp_not_the_next_output(integrator, monkeypatch):
    """Integrate a piecewise constant source across an off-step jump."""
    simulation = _FakeSimulation(integrator=integrator)
    response = _scalar_response(integrator, rate=0)
    response.geometry.main_field = SimpleNamespace(kind="radial")
    simulation.geometry = response.geometry
    simulation.results.input_series.iter_intervals = lambda start, stop, **_kwargs: iter(
        [(start, 0.15, {"source": 1}), (0.15, stop, {"source": 2}), (stop, stop, {"source": 2})]
    )
    evolution = _TimeEvolution(simulation)
    monkeypatch.setattr(
        evolution,
        "_response_and_forcing",
        lambda entries: (response, np.array([entries["source"]]), np.zeros(1)),
    )
    samples = []
    monkeypatch.setattr(
        evolution,
        "_record_output_snapshot",
        lambda key, Br, E, *_, time: samples.extend(
            zip(
                np.atleast_1d(time),
                np.asarray(Br).reshape(-1),
                np.full(np.size(time), E[0]),
                strict=True,
            )
        ),
    )
    evolution.evolve_to_time(
        0.45,
        dt=0.1 if integrator == "euler" else None,
        output_times=[0.15, 0.45],
        quiet=True,
        initialize_from_equilibrium=False,
    )
    np.testing.assert_allclose(samples, [[0.15, 0.15, 2], [0.45, 0.75, 2]], atol=1e-14)


@pytest.mark.requires_jax
@pytest.mark.parametrize("backend", ["numpy"], ids=["backend=numpy"])
@pytest.mark.parametrize("data_source", ["fallback"], ids=["data=fallback"])
def test_explicit_jax_forcing_keeps_exponential_on_device(backend, data_source, monkeypatch):
    """Explicit JAX arrays override the default NumPy backend."""
    import jax
    import jax.numpy as jnp

    def unexpected_transfer(values):
        pytest.fail("The exponential path must not transfer to NumPy.")

    monkeypatch.setattr(linear_evolution, "to_numpy", unexpected_transfer)
    result = induction.evolve_induced_Br(
        _scalar_response("exponential"), jnp.array([2.0]), 0.3, jnp.ones(1)
    )
    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(result, 1 + np.exp(-0.3), atol=1e-14)


@pytest.mark.parametrize("integrator", ["euler", "exponential", "RK45"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_identical_input_records_do_not_restart_accepted_steps(integrator, duplicate, monkeypatch):
    """Repeated input rows are not physical changes."""
    from kompe import SHBasis
    from kompe.coefficients import CoefficientSpace

    from pynamit.storage.field_time_series import FieldTimeSeries

    space = CoefficientSpace(SHBasis(1, 0))
    series = FieldTimeSeries({"source": space}, {"source": ("value",)})
    for time in [0, 0.15] if duplicate else [0]:
        series.add_entry("source", {"value": np.ones(space.shape)}, time)
    simulation = _FakeSimulation(integrator=integrator)
    simulation.results.input_series = series
    response = _scalar_response(integrator)
    evolution = _TimeEvolution(simulation)
    monkeypatch.setattr(
        evolution,
        "_response_and_forcing",
        lambda entries: (response, entries["source"]["value"], np.zeros(1)),
    )
    samples = []
    monkeypatch.setattr(
        evolution, "_record_output_snapshot", lambda key, Br, *_, time: samples.append(Br)
    )
    evolution.evolve_to_time(
        0.3,
        dt=0.1 if integrator == "euler" else None,
        output_times=[0.3],
        initialize_from_equilibrium=False,
        quiet=True,
        rtol=1e-10,
        atol=1e-12,
    )
    expected = 1 - 0.9**3 if integrator == "euler" else 1 - np.exp(-0.3)
    np.testing.assert_allclose(samples, [[expected]], rtol=1e-9)


@pytest.mark.parametrize("jump", [0.3, 3 * 0.1])
def test_exponential_jump_has_no_output_step_delay_or_anticipation(jump, monkeypatch):
    """Sample both sides of a jump without moving its activation."""
    from kompe import SHBasis
    from kompe.coefficients import CoefficientSpace

    from pynamit.storage.field_time_series import FieldTimeSeries

    space = CoefficientSpace(SHBasis(1, 0))
    series = FieldTimeSeries({"source": space}, {"source": ("value",)})
    for time, value in [(0, 1), (jump, 2)]:
        series.add_entry("source", {"value": np.full(space.shape, value)}, time)
    simulation = _FakeSimulation(integrator="exponential")
    simulation.results.input_series = series
    response = _scalar_response("exponential")
    evolution = _TimeEvolution(simulation)
    monkeypatch.setattr(
        evolution,
        "_response_and_forcing",
        lambda entries: (response, entries["source"]["value"], np.zeros(1)),
    )
    samples = []
    monkeypatch.setattr(
        evolution,
        "_record_output_snapshot",
        lambda key, Br, E, *_, time: samples.extend(
            zip(
                np.atleast_1d(time),
                np.asarray(Br).reshape(-1),
                np.full(np.size(time), E[0]),
                strict=True,
            )
        ),
    )
    times = np.array([0.3 - 0.5e-6, 0.3, 0.3 + 0.5e-6, 0.4])
    evolution.evolve_to_time(
        0.4, output_times=times, initialize_from_equilibrium=False, quiet=True
    )
    before = 1 - np.exp(-times)
    after = 2 + ((1 - np.exp(-0.3)) - 2) * np.exp(-(times - 0.3))
    samples = np.asarray(samples)
    np.testing.assert_array_equal(samples[:, 0], times)
    np.testing.assert_allclose(samples[:, 1], np.where(times < 0.3, before, after), atol=1e-14)
    np.testing.assert_array_equal(samples[:, 2], [1, 2, 2, 2])
