"""Read-only access to persisted PynaMIT results."""

from pynamit.results.input_projection import evaluate_projected_input
from pynamit.results.output_fields import evaluate_simulation_output
from pynamit.results.simulation_results import SimulationResults

__all__ = ["SimulationResults", "evaluate_projected_input", "evaluate_simulation_output"]
