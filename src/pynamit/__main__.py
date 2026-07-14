"""Main entry point for running the Pynamit simulation.

This script runs the standard simulation workflow.
"""

from .simulation.workflows.standard import run_pynamit


if __name__ == "__main__":
    run_pynamit()
