"""Execute GUI workflows in an isolated Python process.

The Panel event loop remains responsive, and provider/backend state is
not shared between concurrent simulations. Scientific calculations stay
in the ordinary workflow functions, which scripts can call directly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys


def main(argv=None):
    """Read parameters from stdin and run one requested workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("prepare", "simulate", "movie"))
    parser.add_argument("--backend", choices=("numpy", "jax"), required=True)
    args = parser.parse_args(argv)

    from kompe.math import set_backend

    set_backend(args.backend)
    parameters = json.load(sys.stdin)
    if args.workflow == "prepare":
        from pynamit.workflows.example_inputs import prepare_example_inputs

        parameters["event_time"] = datetime.datetime.fromisoformat(parameters["event_time"])
        prepare_example_inputs(**parameters)
    elif args.workflow == "simulate":
        from pynamit.workflows.prepared_inputs import run_from_inputs

        run_from_inputs(**parameters)
    else:
        from pynamit.plotting import FigureSettings, save_movie

        settings = FigureSettings.from_dict(parameters["settings"])
        save_movie(settings, parameters["output_path"])


if __name__ == "__main__":
    main()
