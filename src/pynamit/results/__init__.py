"""Read-only access to persisted PynaMIT results."""

from pynamit.results.input_projection import evaluate_projected_input
from pynamit.results.simulation_results import SimulationResults

__all__ = ["SimulationResults", "evaluate_projected_input"]
