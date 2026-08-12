"""End-to-end equivalence of native and fallback empirical inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pynamit.external_inputs import get_input_source, native_inputs_available, set_input_source
from pynamit.results.input_projection import evaluate_projected_input
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.workflows import example_inputs as example_inputs_module
from pynamit.workflows.example_inputs import prepare_example_inputs
from tests import SINGLE_PRECISION_REGRESSION_RTOL

_INPUT_KEYS = ("conductance", "boundary_jr", "u")
_RTOL = SINGLE_PRECISION_REGRESSION_RTOL


def _assert_close(name: str, native: Any, fallback: Any) -> None:
    """Compare whole fields with relative L2 and L-infinity errors."""
    native_array = np.asarray(native)
    fallback_array = np.asarray(fallback)
    if native_array.shape != fallback_array.shape:
        pytest.fail(
            f"{name}: native shape {native_array.shape} != fallback shape {fallback_array.shape}"
        )
    difference = np.abs(native_array - fallback_array)
    fallback_norm = float(np.linalg.norm(fallback_array))
    difference_norm = float(np.linalg.norm(native_array - fallback_array))
    fallback_max_abs = float(np.max(np.abs(fallback_array)))
    difference_max_abs = float(np.max(difference))
    relative_l2 = 0.0 if difference_norm == 0.0 else float("inf")
    relative_linf = 0.0 if difference_max_abs == 0.0 else float("inf")
    if fallback_norm:
        relative_l2 = difference_norm / fallback_norm
    if fallback_max_abs:
        relative_linf = difference_max_abs / fallback_max_abs
    if relative_l2 <= _RTOL and relative_linf <= _RTOL:
        return
    flat_index = int(np.nanargmax(difference))
    index = np.unravel_index(flat_index, difference.shape)
    pytest.fail(
        f"{name}: native and fallback field norms differ; "
        f"relative_l2={relative_l2:.6e}, relative_linf={relative_linf:.6e}, "
        f"native_range=[{np.min(native_array):.6e}, {np.max(native_array):.6e}], "
        f"fallback_range=[{np.min(fallback_array):.6e}, {np.max(fallback_array):.6e}], "
        f"worst_index={index}, abs_difference={difference[index]:.6e}"
    )


def _capture_preparation(monkeypatch, *, source: str, directory, main_field_kind: str, ncs: int):
    """Prepare inputs while retaining the provider-facing arrays."""
    captured: dict[str, dict[str, Any]] = {}
    original_conductance = example_inputs_module.get_conductance_inputs
    original_jr = example_inputs_module.get_jr_inputs
    original_wind = example_inputs_module.get_wind_inputs

    def capture_conductance(*args, **kwargs):
        hall, pedersen, lat, lon = original_conductance(*args, **kwargs)
        captured["conductance"] = {
            "hall": np.array(hall, copy=True),
            "pedersen": np.array(pedersen, copy=True),
            "lat": np.array(lat, copy=True),
            "lon": np.array(lon, copy=True),
            "request": kwargs["request"],
        }
        return hall, pedersen, lat, lon

    def capture_jr(*args, **kwargs):
        jr, lat, lon = original_jr(*args, **kwargs)
        captured["boundary_jr"] = {
            "jr": np.array(jr, copy=True),
            "lat": np.array(lat, copy=True),
            "lon": np.array(lon, copy=True),
            "request": kwargs["request"],
        }
        return jr, lat, lon

    def capture_wind(*args, **kwargs):
        result = original_wind(*args, **kwargs)
        if result is None:
            raise RuntimeError("Native/fallback comparison requires neutral wind input.")
        u_theta, u_phi, lat, lon, weights = result
        captured["u"] = {
            "u_theta": np.array(u_theta, copy=True),
            "u_phi": np.array(u_phi, copy=True),
            "lat": np.array(lat, copy=True),
            "lon": np.array(lon, copy=True),
            "weights": None if weights is None else np.array(weights, copy=True),
            "request": kwargs["request"],
        }
        return result

    with monkeypatch.context() as patch:
        patch.setattr(example_inputs_module, "get_conductance_inputs", capture_conductance)
        patch.setattr(example_inputs_module, "get_jr_inputs", capture_jr)
        patch.setattr(example_inputs_module, "get_wind_inputs", capture_wind)
        set_input_source(source)
        simulation = prepare_example_inputs(
            directory,
            final_time=0.0,
            Nmax=4,
            Mmax=4,
            Ncs=ncs,
            main_field_kind=main_field_kind,
            use_boundary_jr=True,
            use_wind=True,
            artifact_storage="netcdf",
        )

    if set(captured) != set(_INPUT_KEYS):
        raise RuntimeError(f"Preparation captured {sorted(captured)}, expected {_INPUT_KEYS}.")
    return simulation, captured


def _coefficient_entries(simulation) -> dict[str, dict[str, np.ndarray]]:
    """Return stored coefficient rows for all compared inputs."""
    result = {}
    for key in _INPUT_KEYS:
        entry = simulation.data.input_series.get_entry(key, 0.0, interpolation=False)
        if entry is None:
            raise RuntimeError(f"Prepared simulation has no {key!r} entry at t=0.")
        result[key] = {name: np.asarray(values) for name, values in entry.items()}
    return result


def _synthesized_values(simulation) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate compared inputs on the simulation model grid."""
    return {
        key: {
            name: np.asarray(values)
            for name, values in evaluate_projected_input(
                simulation,
                key,
                0.0,
                grid=simulation.geometry.model_grid,
                interpolation=False,
                include_derived=True,
            ).items()
        }
        for key in _INPUT_KEYS
    }


def _assert_mappings_close(
    stage: str, native: dict[str, np.ndarray], fallback: dict[str, np.ndarray]
) -> None:
    """Compare mappings with identical keys and named diagnostics."""
    assert set(native) == set(fallback), f"{stage}: fields differ"
    for name in sorted(native):
        _assert_close(f"{stage}:{name}", native[name], fallback[name])


@pytest.mark.parametrize(
    ("backend", "data_source"), [("numpy", "fallback")], ids=["numpy-native-vs-fallback"]
)
@pytest.mark.parametrize(
    ("main_field_kind", "ncs"),
    [("dipole", 8), ("igrf", 18)],
    ids=["centered-dipole", "geographic"],
)
def test_native_and_fallback_inputs_match_through_projection(
    tmp_path, monkeypatch, backend, data_source, main_field_kind, ncs
):
    """Native and cached inputs remain equivalent through projection."""
    del backend, data_source
    if not native_inputs_available():
        pytest.skip("Native empirical-input providers are unavailable.")

    previous_source = get_input_source()
    try:
        native_simulation, native_raw = _capture_preparation(
            monkeypatch,
            source="native",
            directory=tmp_path / "native",
            main_field_kind=main_field_kind,
            ncs=ncs,
        )
        fallback_simulation, fallback_raw = _capture_preparation(
            monkeypatch,
            source="fallback",
            directory=tmp_path / "fallback",
            main_field_kind=main_field_kind,
            ncs=ncs,
        )
    finally:
        set_input_source(previous_source)

    native_request = native_raw["conductance"]["request"]
    fallback_request = fallback_raw["conductance"]["request"]
    assert native_request.source_grid.coordinate_contract == (
        fallback_request.source_grid.coordinate_contract
    )
    assert native_request.model_grid.coordinate_contract == (
        fallback_request.model_grid.coordinate_contract
    )
    assert native_request.model_epoch == pytest.approx(fallback_request.model_epoch)
    for view in ("source_grid", "model_grid"):
        native_grid = getattr(native_request, view)
        fallback_grid = getattr(fallback_request, view)
        assert native_grid.coordinate_identity == fallback_grid.coordinate_identity

    for request, raw_inputs in ((native_request, native_raw), (fallback_request, fallback_raw)):
        source_grid = request.source_grid
        for key in _INPUT_KEYS:
            assert raw_inputs[key]["request"] is request
            returned_identity = source_grid.coordinate_contract.coordinate_identity(
                raw_inputs[key]["lat"], raw_inputs[key]["lon"]
            )
            assert returned_identity == source_grid.coordinate_identity

    _assert_mappings_close(
        "provider:conductance",
        {name: native_raw["conductance"][name] for name in ("hall", "pedersen")},
        {name: fallback_raw["conductance"][name] for name in ("hall", "pedersen")},
    )
    _assert_mappings_close(
        "provider:boundary_jr",
        {"jr": native_raw["boundary_jr"]["jr"]},
        {"jr": fallback_raw["boundary_jr"]["jr"]},
    )
    _assert_mappings_close(
        "provider:u",
        {name: native_raw["u"][name] for name in ("u_theta", "u_phi")},
        {name: fallback_raw["u"][name] for name in ("u_theta", "u_phi")},
    )

    native_log = ionospheric_closure.conductance_to_log_coordinates(
        native_raw["conductance"]["pedersen"], native_raw["conductance"]["hall"]
    )
    fallback_log = ionospheric_closure.conductance_to_log_coordinates(
        fallback_raw["conductance"]["pedersen"], fallback_raw["conductance"]["hall"]
    )
    _assert_close("canonical:log_magnitude", native_log[0], fallback_log[0])
    _assert_close("canonical:log_ratio", native_log[1], fallback_log[1])

    native_coefficients = _coefficient_entries(native_simulation)
    fallback_coefficients = _coefficient_entries(fallback_simulation)
    native_synthesized = _synthesized_values(native_simulation)
    fallback_synthesized = _synthesized_values(fallback_simulation)
    for key in _INPUT_KEYS:
        _assert_mappings_close(
            f"coefficients:{key}", native_coefficients[key], fallback_coefficients[key]
        )
        _assert_mappings_close(
            f"synthesized:{key}", native_synthesized[key], fallback_synthesized[key]
        )
