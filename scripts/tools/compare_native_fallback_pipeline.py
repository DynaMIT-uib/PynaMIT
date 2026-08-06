#!/usr/bin/env python3
"""Compare native and fallback inputs through PynaMIT.

This diagnostic uses PynaMIT's existing input machinery:

1. ``prepare_pynamit_inputs`` obtains provider samples and calls
   the normal ``Simulation.set_*`` input methods.
2. The input pipeline converts conductance to canonical logarithmic
   coordinates and projects fields into their storage bases.
3. Stored coefficient rows are read from ``FieldTimeSeries``.
4. ``evaluate_projected_input`` synthesizes coefficients on the
   model grid and reconstructs physical conductance values.

Native and fallback executions are compared at four boundaries:

* provider samples;
* canonical pre-projection values;
* projected and stored coefficients;
* synthesized values on the model grid.

The default cases match the three input-projection configurations
used by the failing non-wind regression tests:

* Nmax=12, Mmax=12, Ncs=22;
* Nmax=10, Mmax=10, Ncs=20;
* Nmax=10, Mmax=8, Ncs=18.

Examples
--------
Run the failing-test profiles and write a CI artifact::

    python scripts/tools/compare_native_fallback_pipeline.py \
        --json native-fallback-pipeline.json

Collect diagnostics without failing the workflow::

    python scripts/tools/compare_native_fallback_pipeline.py \
        --no-fail --json native-fallback-pipeline.json

Run all source grids in the fallback collection. This uses the
Nmax=Mmax=4 space from the fallback-generation utility::

    python scripts/tools/compare_native_fallback_pipeline.py \
        --all-fallback-grids --json all-native-fallback-pipeline.json
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pynamit.external_inputs import (
    _load_fallback,
    get_input_source,
    native_inputs_available,
    set_input_source,
)
from pynamit.simulation.electrodynamics import ionospheric_closure
from pynamit.simulation.workflows import prepared_inputs as prepared_inputs_module
from pynamit.simulation.workflows.prepared_inputs import prepare_pynamit_inputs
from pynamit.visualization.input_projection import evaluate_projected_input


SCRIPT_VERSION = "2026-08-06.3"


@dataclass(frozen=True)
class ProjectionCase:
    """One reproducible input-projection configuration."""

    name: str
    Nmax: int
    Mmax: int
    Ncs: int
    main_field_kind: str = "dipole"
    main_field_epoch: float = 2020.0
    fallback_grid_id: str | None = None


FAILING_TEST_CASES = (
    ProjectionCase(
        name="dipole-n12-m12-ncs22",
        Nmax=12,
        Mmax=12,
        Ncs=22,
        fallback_grid_id="centered-dipole-2020-ncs-22",
    ),
    ProjectionCase(
        name="dipole-n10-m10-ncs20",
        Nmax=10,
        Mmax=10,
        Ncs=20,
        fallback_grid_id="centered-dipole-2020-ncs-20",
    ),
    ProjectionCase(
        name="dipole-n10-m08-ncs18",
        Nmax=10,
        Mmax=8,
        Ncs=18,
        fallback_grid_id="centered-dipole-2020-ncs-18",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in FAILING_TEST_CASES],
        default=[],
        help="Run only this predefined failing-test profile. May be repeated.",
    )
    parser.add_argument(
        "--all-fallback-grids",
        action="store_true",
        help=(
            "Run every cached fallback source grid with Nmax=Mmax=4, matching "
            "the coefficient space used by regenerate_fallback_inputs.py."
        ),
    )
    parser.add_argument(
        "--raw-rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance for provider and canonical samples (default: 1e-10).",
    )
    parser.add_argument(
        "--coefficient-rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance for stored projected coefficients (default: 1e-10).",
    )
    parser.add_argument(
        "--synthesized-rtol",
        type=float,
        default=1e-10,
        help="Relative tolerance for synthesized model-grid fields (default: 1e-10).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance used at all non-coordinate stages (default: 0).",
    )
    parser.add_argument(
        "--artifact-storage",
        choices=("auto", "netcdf", "zarr"),
        default="auto",
        help="Storage backend used by temporary prepared-input packages.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a machine-readable report.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Always exit successfully after printing and writing diagnostics.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    """Return JSON-compatible forms for NumPy and path-like values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _event_time(value: Any) -> Any:
    if value is None or isinstance(value, (dt.datetime, dt.date)):
        return value
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite_count": int(np.count_nonzero(finite)),
        "sha256": _sha256_array(array),
    }
    if array.size:
        result.update(
            minimum=float(np.nanmin(array)),
            maximum=float(np.nanmax(array)),
            sum=float(np.nansum(array)),
            l2_norm=float(np.linalg.norm(np.nan_to_num(array))),
        )
    return result


def _scaled_error(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> np.ndarray:
    difference = np.abs(candidate - reference)
    denominator = atol + rtol * np.abs(reference)
    return np.divide(
        difference,
        denominator,
        out=np.where(difference == 0.0, 0.0, np.inf),
        where=denominator > 0.0,
    )


def _compare_array(
    *,
    case: ProjectionCase,
    stage: str,
    field: str,
    native: Any,
    fallback: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    native_array = np.asarray(native)
    fallback_array = np.asarray(fallback)
    result: dict[str, Any] = {
        "case": case.name,
        "stage": stage,
        "field": field,
        "rtol": rtol,
        "atol": atol,
        "same_shape": native_array.shape == fallback_array.shape,
        "native": _array_summary(native_array),
        "fallback": _array_summary(fallback_array),
    }

    if native_array.shape != fallback_array.shape:
        result.update(passed=False, exact_equal=False, reason="shape mismatch")
        return result

    difference = native_array - fallback_array
    absolute_difference = np.abs(difference)
    fallback_norm = float(np.linalg.norm(fallback_array))
    difference_norm = float(np.linalg.norm(difference))
    relative_l2 = (
        difference_norm / fallback_norm
        if fallback_norm > 0.0
        else 0.0
        if difference_norm == 0.0
        else float("inf")
    )
    scaled = _scaled_error(native_array, fallback_array, rtol=rtol, atol=atol)
    worst_flat = int(np.argmax(scaled)) if scaled.size else 0
    worst_index = (
        [int(index) for index in np.unravel_index(worst_flat, scaled.shape)]
        if scaled.size
        else []
    )

    if scaled.size:
        index = tuple(worst_index)
        worst_native = float(native_array[index])
        worst_fallback = float(fallback_array[index])
        worst_abs = float(absolute_difference[index])
        max_scaled = float(scaled[index])
    else:
        worst_native = worst_fallback = worst_abs = max_scaled = 0.0

    result.update(
        passed=bool(
            np.allclose(
                native_array,
                fallback_array,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        ),
        exact_equal=bool(np.array_equal(native_array, fallback_array)),
        max_abs_difference=(
            float(np.max(absolute_difference)) if absolute_difference.size else 0.0
        ),
        l2_difference=difference_norm,
        relative_l2_difference=relative_l2,
        max_scaled_error=max_scaled,
        worst_index=worst_index,
        worst_native=worst_native,
        worst_fallback=worst_fallback,
        worst_abs_difference=worst_abs,
    )
    return result


def _print_result(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"{status:4}  {result['case']:25}  {result['stage']:15}  "
        f"{result['field']:32}  "
        f"rel_l2={result.get('relative_l2_difference', float('nan')):.6e}  "
        f"max_abs={result.get('max_abs_difference', float('nan')):.6e}  "
        f"scaled={result.get('max_scaled_error', float('nan')):.6e}"
    )
    if result["passed"]:
        return
    if result.get("reason"):
        print(f"      reason={result['reason']}")
        return
    print(
        "      "
        f"worst_index={tuple(result['worst_index'])} "
        f"native={result['worst_native']:.17e} "
        f"fallback={result['worst_fallback']:.17e} "
        f"abs_diff={result['worst_abs_difference']:.6e}"
    )
    print(f"      native_sha256={result['native']['sha256']}")
    print(f"      fallback_sha256={result['fallback']['sha256']}")


def _distribution_metadata(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}

    direct_url = None
    with contextlib.suppress(Exception):
        text = distribution.read_text("direct_url.json")
        if text:
            direct_url = json.loads(text)

    return {
        "installed": True,
        "version": distribution.version,
        "direct_url": direct_url,
    }


def _pyamps_coefficient_metadata() -> dict[str, Any] | None:
    try:
        import pyamps
    except Exception:
        return None

    path = (
        Path(pyamps.__file__).resolve().parent
        / "coefficients"
        / "SW_OPER_MIO_SHA_2E_00000000T000000_99999999T999999_0104.txt"
    )
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _environment_report() -> dict[str, Any]:
    numpy_config = io.StringIO()
    with contextlib.redirect_stdout(numpy_config):
        np.show_config()

    packages = (
        "numpy",
        "scipy",
        "apexpy",
        "pyamps",
        "lompe",
        "kompe",
        "pynamit",
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "executable": sys.executable,
        "input_source_at_start": get_input_source(),
        "packages": {name: _distribution_metadata(name) for name in packages},
        "pyamps_coefficient_file": _pyamps_coefficient_metadata(),
        "numpy_config": numpy_config.getvalue(),
        "selected_environment_variables": {
            name: os.environ.get(name)
            for name in (
                "PYNAMIT_INPUT_SOURCE",
                "KOMPE_LEAST_SQUARES_SOLVER",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
            if name in os.environ
        },
    }


@contextmanager
def _capture_prepared_provider_inputs() -> Iterator[dict[str, Any]]:
    """Capture provider arrays consumed by input preparation.

    The workflow imports provider functions into its module namespace.
    Wrapping those names observes the exact request and return values
    without reimplementing coordinate construction or projection.
    """

    captured: dict[str, Any] = {}
    original_conductance = prepared_inputs_module.get_conductance_inputs
    original_jr = prepared_inputs_module.get_jr_inputs

    def capture_conductance(*args, **kwargs):
        result = original_conductance(*args, **kwargs)
        hall, pedersen, lat, lon = result
        captured["conductance"] = {
            "hall": np.array(hall, copy=True),
            "pedersen": np.array(pedersen, copy=True),
            "lat": np.array(lat, copy=True),
            "lon": np.array(lon, copy=True),
            "request": kwargs.get("request"),
        }
        return result

    def capture_jr(*args, **kwargs):
        result = original_jr(*args, **kwargs)
        jr, lat, lon = result
        captured["boundary_jr"] = {
            "jr": np.array(jr, copy=True),
            "lat": np.array(lat, copy=True),
            "lon": np.array(lon, copy=True),
            "request": kwargs.get("request"),
        }
        return result

    prepared_inputs_module.get_conductance_inputs = capture_conductance
    prepared_inputs_module.get_jr_inputs = capture_jr
    try:
        yield captured
    finally:
        prepared_inputs_module.get_conductance_inputs = original_conductance
        prepared_inputs_module.get_jr_inputs = original_jr


def _prepare_one_source(
    case: ProjectionCase,
    *,
    source: str,
    directory: Path,
    artifact_storage: str,
):
    """Prepare one package through the normal workflow."""

    set_input_source(source)
    with _capture_prepared_provider_inputs() as captured:
        simulation = prepare_pynamit_inputs(
            input_directory=directory,
            final_time=0.0,
            Nmax=case.Nmax,
            Mmax=case.Mmax,
            Ncs=case.Ncs,
            main_field_kind=case.main_field_kind,
            main_field_epoch=case.main_field_epoch,
            use_wind=False,
            use_Q_eff=False,
            use_boundary_jr=True,
            multi_data=False,
            artifact_storage=artifact_storage,
        )

    missing = {"conductance", "boundary_jr"} - set(captured)
    if missing:
        raise RuntimeError(
            f"{source} preparation did not expose expected provider inputs: "
            + ", ".join(sorted(missing))
        )
    return simulation, captured


def _coefficient_entries(simulation) -> dict[str, dict[str, np.ndarray]]:
    """Read the stored coefficient rows through FieldTimeSeries."""

    entries: dict[str, dict[str, np.ndarray]] = {}
    for key in ("conductance", "boundary_jr"):
        entry = simulation.run_data.input_series.get_entry(
            key, 0.0, interpolation=False
        )
        if entry is None:
            raise RuntimeError(f"Prepared simulation has no {key!r} entry at t=0.")
        entries[key] = {
            variable: np.array(values, copy=True)
            for variable, values in entry.items()
        }
    return entries


def _synthesized_values(simulation) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate coefficients with PynaMIT's inspection machinery."""

    return {
        "conductance": {
            key: np.asarray(value)
            for key, value in evaluate_projected_input(
                simulation,
                "conductance",
                0.0,
                grid=simulation.geometry.model_grid,
                interpolation=False,
                include_derived=True,
            ).items()
        },
        "boundary_jr": {
            key: np.asarray(value)
            for key, value in evaluate_projected_input(
                simulation,
                "boundary_jr",
                0.0,
                grid=simulation.geometry.model_grid,
                interpolation=False,
                include_derived=True,
            ).items()
        },
    }


def _canonical_conductance(captured: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return the conductance variables used by the projection."""

    values = captured["conductance"]
    log_magnitude, log_ratio = ionospheric_closure.conductance_to_log_coordinates(
        values["pedersen"], values["hall"]
    )
    return {
        "log_conductance_magnitude": np.asarray(log_magnitude),
        "log_hall_to_pedersen_ratio": np.asarray(log_ratio),
    }


def _cases_from_fallback() -> tuple[ProjectionCase, ...]:
    collection = _load_fallback()
    cases = []
    for grid_id, dataset in sorted(collection.datasets["conductance"].items()):
        source_grid = dataset.source_grid
        geometry = source_grid.sampling_geometry
        origin = source_grid.provenance.get("originating_model_frame", {})
        try:
            ncs = int(geometry["ncs"])
            main_field_kind = str(origin["main_field_kind"])
            epoch = float(origin["epoch"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Fallback grid {grid_id!r} lacks projection reconstruction metadata."
            ) from exc
        cases.append(
            ProjectionCase(
                name=f"fallback-grid-{grid_id}",
                Nmax=4,
                Mmax=4,
                Ncs=ncs,
                main_field_kind=main_field_kind,
                main_field_epoch=epoch,
                fallback_grid_id=grid_id,
            )
        )
    return tuple(cases)


def _selected_cases(args: argparse.Namespace) -> tuple[ProjectionCase, ...]:
    if args.all_fallback_grids and args.case:
        raise SystemExit("--all-fallback-grids cannot be combined with --case.")
    if args.all_fallback_grids:
        return _cases_from_fallback()
    if args.case:
        selected = set(args.case)
        return tuple(case for case in FAILING_TEST_CASES if case.name in selected)
    return FAILING_TEST_CASES


def _compare_mapping(
    *,
    case: ProjectionCase,
    stage: str,
    native: dict[str, Any],
    fallback: dict[str, Any],
    rtol: float,
    atol: float,
    report: list[dict[str, Any]],
) -> None:
    native_keys = set(native)
    fallback_keys = set(fallback)
    if native_keys != fallback_keys:
        missing_native = sorted(fallback_keys - native_keys)
        missing_fallback = sorted(native_keys - fallback_keys)
        raise RuntimeError(
            f"{case.name} {stage} keys differ; "
            f"missing_native={missing_native}, missing_fallback={missing_fallback}"
        )
    for field in sorted(native):
        result = _compare_array(
            case=case,
            stage=stage,
            field=field,
            native=native[field],
            fallback=fallback[field],
            rtol=rtol,
            atol=atol,
        )
        report.append(result)
        _print_result(result)


def _run_case(
    case: ProjectionCase,
    *,
    root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(
        f"\n=== {case.name}: Nmax={case.Nmax}, Mmax={case.Mmax}, "
        f"Ncs={case.Ncs}, main_field={case.main_field_kind}, "
        f"epoch={case.main_field_epoch} ==="
    )

    native_simulation, native_raw = _prepare_one_source(
        case,
        source="native",
        directory=root / case.name / "native",
        artifact_storage=args.artifact_storage,
    )
    fallback_simulation, fallback_raw = _prepare_one_source(
        case,
        source="fallback",
        directory=root / case.name / "fallback",
        artifact_storage=args.artifact_storage,
    )

    comparisons: list[dict[str, Any]] = []

    # The coordinate contract should be bitwise identical.
    raw_coordinates_native = {
        "conductance_lat": native_raw["conductance"]["lat"],
        "conductance_lon": native_raw["conductance"]["lon"],
        "boundary_jr_lat": native_raw["boundary_jr"]["lat"],
        "boundary_jr_lon": native_raw["boundary_jr"]["lon"],
        "model_lat": native_simulation.geometry.model_grid.lat,
        "model_lon": native_simulation.geometry.model_grid.lon,
    }
    raw_coordinates_fallback = {
        "conductance_lat": fallback_raw["conductance"]["lat"],
        "conductance_lon": fallback_raw["conductance"]["lon"],
        "boundary_jr_lat": fallback_raw["boundary_jr"]["lat"],
        "boundary_jr_lon": fallback_raw["boundary_jr"]["lon"],
        "model_lat": fallback_simulation.geometry.model_grid.lat,
        "model_lon": fallback_simulation.geometry.model_grid.lon,
    }
    _compare_mapping(
        case=case,
        stage="coordinates",
        native=raw_coordinates_native,
        fallback=raw_coordinates_fallback,
        rtol=0.0,
        atol=0.0,
        report=comparisons,
    )

    raw_native = {
        "hall": native_raw["conductance"]["hall"],
        "pedersen": native_raw["conductance"]["pedersen"],
        "boundary_jr": native_raw["boundary_jr"]["jr"],
    }
    raw_fallback = {
        "hall": fallback_raw["conductance"]["hall"],
        "pedersen": fallback_raw["conductance"]["pedersen"],
        "boundary_jr": fallback_raw["boundary_jr"]["jr"],
    }
    _compare_mapping(
        case=case,
        stage="provider_raw",
        native=raw_native,
        fallback=raw_fallback,
        rtol=args.raw_rtol,
        atol=args.atol,
        report=comparisons,
    )

    canonical_native = _canonical_conductance(native_raw)
    canonical_fallback = _canonical_conductance(fallback_raw)
    _compare_mapping(
        case=case,
        stage="canonical",
        native=canonical_native,
        fallback=canonical_fallback,
        rtol=args.raw_rtol,
        atol=args.atol,
        report=comparisons,
    )

    native_coefficients = _coefficient_entries(native_simulation)
    fallback_coefficients = _coefficient_entries(fallback_simulation)
    for key in ("conductance", "boundary_jr"):
        _compare_mapping(
            case=case,
            stage=f"coefficients:{key}",
            native=native_coefficients[key],
            fallback=fallback_coefficients[key],
            rtol=args.coefficient_rtol,
            atol=args.atol,
            report=comparisons,
        )

    native_synthesized = _synthesized_values(native_simulation)
    fallback_synthesized = _synthesized_values(fallback_simulation)
    for key in ("conductance", "boundary_jr"):
        _compare_mapping(
            case=case,
            stage=f"synthesized:{key}",
            native=native_synthesized[key],
            fallback=fallback_synthesized[key],
            rtol=args.synthesized_rtol,
            atol=args.atol,
            report=comparisons,
        )

    failed = [result for result in comparisons if not result["passed"]]
    stage_max_relative_l2: dict[str, float] = {}
    for result in comparisons:
        value = result.get("relative_l2_difference")
        if value is None:
            continue
        stage = result["stage"]
        stage_max_relative_l2[stage] = max(
            stage_max_relative_l2.get(stage, 0.0), float(value)
        )

    print(f"--- {case.name}: {len(failed)} failure(s) / {len(comparisons)} comparisons")
    return {
        "case": asdict(case),
        "passed": not failed,
        "failure_count": len(failed),
        "comparison_count": len(comparisons),
        "stage_max_relative_l2": stage_max_relative_l2,
        "comparisons": comparisons,
    }


def main() -> int:
    """Run the native and fallback pipeline comparison."""
    args = _parse_args()
    print(f"PynaMIT native/fallback pipeline comparator {SCRIPT_VERSION}")
    for name in ("raw_rtol", "coefficient_rtol", "synthesized_rtol", "atol"):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative.")

    if not native_inputs_available():
        raise SystemExit(
            "Native input providers are unavailable. Install PynaMIT's input "
            "dependencies before running this comparison."
        )

    cases = _selected_cases(args)
    fallback = _load_fallback()
    report: dict[str, Any] = {
        "event_time": str(_event_time(fallback.event_time)),
        "fallback_schema_version": fallback.version,
        "tolerances": {
            "raw_rtol": args.raw_rtol,
            "coefficient_rtol": args.coefficient_rtol,
            "synthesized_rtol": args.synthesized_rtol,
            "atol": args.atol,
        },
        "environment": _environment_report(),
        "cases": [],
    }

    previous_source = get_input_source()
    try:
        with tempfile.TemporaryDirectory(prefix="pynamit-input-comparison-") as root:
            root_path = Path(root)
            for case in cases:
                report["cases"].append(_run_case(case, root=root_path, args=args))
    finally:
        set_input_source(previous_source)

    report["passed"] = all(case["passed"] for case in report["cases"])
    report["failure_count"] = sum(case["failure_count"] for case in report["cases"])
    report["comparison_count"] = sum(
        case["comparison_count"] for case in report["cases"]
    )

    print(
        f"\nCompared {len(report['cases'])} projection case(s), "
        f"{report['comparison_count']} arrays: "
        f"{report['failure_count']} failure(s)."
    )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {args.json}")

    return 0 if args.no_fail or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
