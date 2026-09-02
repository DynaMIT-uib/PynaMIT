"""Reusable input preparation and simulation workflows."""

from pynamit.workflows.example import run_example
from pynamit.workflows.example_inputs import prepare_example_inputs
from pynamit.workflows.prepared_inputs import run_from_inputs

__all__ = ["prepare_example_inputs", "run_example", "run_from_inputs"]
